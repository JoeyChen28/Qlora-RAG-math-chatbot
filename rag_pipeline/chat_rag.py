import argparse
import json
import re
import warnings
from pathlib import Path

import faiss

# 默认向量库根目录：与本文件同级的 rag_pipeline/data（内含 vector_db/ 或 RAG/）
DEFAULT_RAG_DATA_ROOT = str(Path(__file__).resolve().parent / "data")
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RAG chat (QLoRA + FAISS) with your existing indices")

    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional LoRA/QLoRA adapter path. If not set, runs base model only.",
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization for base model loading")

    parser.add_argument(
        "--rag_root",
        type=str,
        default=DEFAULT_RAG_DATA_ROOT,
        help="Data root: looks for vector_db/ or RAG/ under this path (default: rag_pipeline/data).",
    )
    parser.add_argument(
        "--rag_variant",
        type=str,
        choices=["ece269"],
        default="ece269",
        help="Which pre-built index/meta pair to use (no rebuilding needed).",
    )
    parser.add_argument("--index_file", type=str, default=None, help="Override FAISS index file path")
    parser.add_argument("--meta_file", type=str, default=None, help="Override metadata json file path")

    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--query_instruction", type=str, default="Represent this sentence for searching relevant passages: ")

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful and concise mathematical assistant. Answer the user's question using ONLY the retrieved context below.\nRule 1: Be direct and concise. Do not repeat the same point.\nRule 2: Do not output any proof-ending symbols.\nRule 3: Do not output source labels like [Source 1] in the text.\n\nContext:\n{context}",
        help="System prompt template. Use {context} placeholder for retrieved context.",
    )
    parser.add_argument("--keep_think", action="store_true", default=False, help="Keep <think>...</think> blocks in output")
    parser.add_argument("--show_sources", action="store_true", help="Print retrieved chunk sources and short excerpts")

    return parser.parse_args()


def resolve_index_meta(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.index_file and args.meta_file:
        return Path(args.index_file), Path(args.meta_file)

    variant = getattr(args, "rag_variant", "ece269")
    user_root = Path(args.rag_root)
    pkg_data = Path(__file__).resolve().parent / "data"

    search_roots: list[Path] = []
    seen_roots: set[str] = set()
    for r in (user_root, pkg_data):
        key = str(r.resolve())
        if key not in seen_roots:
            seen_roots.add(key)
            search_roots.append(Path(key))

    subdirs = ("vector_db", "RAG")
    tried: list[tuple[Path, Path]] = []
    for root in search_roots:
        for sub in subdirs:
            index_file = root / sub / f"{variant}.index"
            meta_file = root / sub / f"{variant}_meta.json"
            tried.append((index_file, meta_file))
            if index_file.is_file() and meta_file.is_file():
                return index_file, meta_file

    lines = "\n".join(f"  - {idx} + {meta}" for idx, meta in tried)
    raise FileNotFoundError(
        f"No FAISS index + meta found for variant '{variant}'. Tried:\n{lines}\n"
        f"Generate with: cd rag_pipeline/data_preparation && python build_vector_db.py "
        f"(set output paths to .../vector_db/ece269.index and ece269_meta.json)."
    )


def format_source(item: dict) -> str:
    assignment = item.get("assignment", "")
    problem_id = item.get("problem_id", "")
    sub_id = item.get("sub_id", "")
    chunk_type = item.get("chunk_type", "")

    src = assignment if assignment else "UnknownAssignment"
    if problem_id:
        src += f" Q{problem_id}"
    if sub_id and sub_id != "main":
        src += f"({sub_id})"
    if chunk_type:
        src += f" [{chunk_type}]"
    return src


def _llm_input_device(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


def load_retriever(args: argparse.Namespace, index_file: Path, meta_file: Path):
    emb_device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(args.embedding_model, device=emb_device)
    index = faiss.read_index(str(index_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if len(metadata) == 0:
        raise ValueError(f"Empty metadata loaded from {meta_file}")
    return embed_model, index, metadata


def search(
    *,
    query: str,
    args: argparse.Namespace,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    metadata: list,
) -> list[dict]:
    query_text = f"{args.query_instruction}{query}"
    emb = embed_model.encode([query_text], normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")

    scores, indices = index.search(emb, args.top_k)

    results: list[dict] = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0 or i >= len(metadata):
            continue
        item = metadata[i].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def build_messages(args: argparse.Namespace, question: str, retrieved_chunks: list[dict]) -> list[dict]:
    context_blocks = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        source_name = format_source(item)
        block = f"[Source {idx}: {source_name}]\n{item.get('text', '').strip()}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    system_prompt = args.system_prompt.format(context=context_text)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def clean_output(text: str, *, strip_think: bool) -> str:
    text = text.strip()
    unwanted_symbols = ["\\(\\blacksquare\\)", "\\blacksquare", "The final answer is:"]
    for symbol in unwanted_symbols:
        text = text.replace(symbol, "")
    if strip_think:
        # Remove <think>...</think> blocks (including newlines).
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def generate_answer(
    *,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
    dev = _llm_input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    do_sample = args.temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return clean_output(answer, strip_think=not args.keep_think)


def build_rag_args(
    *,
    base_model_name: str = "Qwen/Qwen3-4B",
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
    rag_root: str = DEFAULT_RAG_DATA_ROOT,
    rag_variant: str = "ece269",
    index_file: str | None = None,
    meta_file: str | None = None,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    query_instruction: str = "Represent this sentence for searching relevant passages: ",
    top_k: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.3,
    repetition_penalty: float = 1.1,
    system_prompt: str = (
        "You are a helpful and concise mathematical assistant. Answer the user's question using ONLY the retrieved context below.\n"
        "Rule 1: Be direct and concise. Do not repeat the same point.\n"
        "Rule 2: Do not output any proof-ending symbols.\n"
        "Rule 3: Do not output source labels like [Source 1] in the text.\n\n"
        "Context:\n{context}"
    ),
    keep_think: bool = False,
    show_sources: bool = False,
) -> argparse.Namespace:
    """Build Namespace for RAG without CLI (Streamlit / API)."""
    ns = argparse.Namespace()
    ns.base_model_name = base_model_name
    ns.adapter_path = adapter_path
    ns.load_in_4bit = load_in_4bit
    ns.rag_root = rag_root
    ns.rag_variant = rag_variant
    ns.index_file = index_file
    ns.meta_file = meta_file
    ns.embedding_model = embedding_model
    ns.query_instruction = query_instruction
    ns.top_k = top_k
    ns.max_new_tokens = max_new_tokens
    ns.temperature = temperature
    ns.top_p = top_p
    ns.repetition_penalty = repetition_penalty
    ns.system_prompt = system_prompt
    ns.keep_think = keep_think
    ns.show_sources = show_sources
    return ns


def answer_question_rag(
    question: str,
    *,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    metadata: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_model_name: str = "Qwen/Qwen3-4B",
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
    rag_root: str = DEFAULT_RAG_DATA_ROOT,
    rag_variant: str = "ece269",
    index_file: str | None = None,
    meta_file: str | None = None,
    embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    query_instruction: str = "Represent this sentence for searching relevant passages: ",
    top_k: int = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.3,
    repetition_penalty: float = 1.1,
    system_prompt: str = (
        "You are a helpful and concise mathematical assistant. Answer the user's question using ONLY the retrieved context below.\n"
        "Rule 1: Be direct and concise. Do not repeat the same point.\n"
        "Rule 2: Do not output any proof-ending symbols.\n"
        "Rule 3: Do not output source labels like [Source 1] in the text.\n\n"
        "Context:\n{context}"
    ),
    keep_think: bool = False,
    show_sources: bool = False,
) -> tuple[str, list[dict]]:
    """Single-turn RAG when embedder / index / model are already loaded (e.g. Streamlit cache)."""
    args = build_rag_args(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        rag_root=rag_root,
        rag_variant=rag_variant,
        index_file=index_file,
        meta_file=meta_file,
        embedding_model=embedding_model_name,
        query_instruction=query_instruction,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        system_prompt=system_prompt,
        keep_think=keep_think,
    )
    retrieved = search(
        query=question,
        args=args,
        embed_model=embed_model,
        index=index,
        metadata=metadata,
    )
    messages = build_messages(args, question, retrieved)
    answer = generate_answer(args=args, model=model, tokenizer=tokenizer, messages=messages)
    _ = show_sources
    return answer, retrieved


def load_generator(args: argparse.Namespace) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        try:
            _ = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            cuda_ok = False

    use_4bit = bool(getattr(args, "load_in_4bit", False)) and cuda_ok
    if getattr(args, "load_in_4bit", False) and not cuda_ok:
        warnings.warn(
            "已请求 4-bit 量化，但当前环境 torch.cuda.is_available() 为 False（多为安装了 CPU 版 PyTorch，"
            "或驱动/CUDA 未就绪）。将退回 CPU float32 全精度加载，速度极慢且易表现为“卡住”。"
            "请到 https://pytorch.org 安装带 CUDA 的 torch，并在 nvidia-smi 正常的前提下重试。",
            stacklevel=1,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}

    if use_4bit:
        use_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            **load_kw,
        )
    elif cuda_ok:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            **load_kw,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch.float32,
            device_map=None,
            **load_kw,
        )

    if args.adapter_path:
        adapter_dir = Path(args.adapter_path)
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"未找到 adapter_config.json：{adapter_dir}\n"
                "请将 --adapter_path / 侧边栏 Adapter Path 设为包含 PEFT 配置的目录，"
                "例如本仓库的 Qlora-result/checkpoint-2400 或 checkpoint-5400。"
            )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        model = base_model
    model.eval()

    dev = _llm_input_device(model)
    mode = "4-bit (CUDA)" if use_4bit else ("FP16 (cuda:0)" if cuda_ok else "FP32 (CPU)")
    print(f"[chat_rag] LLM loaded: device={dev}, mode={mode}, cuda_available={torch.cuda.is_available()}")

    return model, tokenizer


def run_chat(args: argparse.Namespace) -> None:
    index_file, meta_file = resolve_index_meta(args)

    print("Loading embedding model...")
    embed_model, index, metadata = load_retriever(args, index_file, meta_file)

    print("Loading tokenizer and generator...")
    model, tokenizer = load_generator(args)

    print("\nRAG chat started.")
    print("Type your question. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exit.")
            break
        if not question:
            continue

        retrieved = search(query=question, args=args, embed_model=embed_model, index=index, metadata=metadata)

        if args.show_sources:
            print("\n" + "=" * 100)
            print("Retrieved Chunks")
            print("=" * 100)
            for i, item in enumerate(retrieved, start=1):
                src = format_source(item)
                text = item.get("text", "").replace("\n", " ")
                snippet = text[:250] + ("..." if len(text) > 250 else "")
                print(f"[{i}] {src} | score={item.get('score', 0):.4f}")
                print(snippet)
                print("-" * 100)

        messages = build_messages(args, question, retrieved)
        answer = generate_answer(args=args, model=model, tokenizer=tokenizer, messages=messages)

        print("\n" + "=" * 100)
        print("Assistant")
        print("=" * 100)
        print(answer)
        print()


def main() -> None:
    args = parse_args()

    # Default adapter path is convenient for your current project layout.
    if args.adapter_path is None:
        default_adapter = Path(__file__).resolve().parent.parent / "Qlora-result" / "checkpoint-2400"
        if default_adapter.exists():
            args.adapter_path = str(default_adapter)

    run_chat(args)


if __name__ == "__main__":
    main()

