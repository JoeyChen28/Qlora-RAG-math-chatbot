
# Qlora-RAG-math-chatbot
<img width="1912" height="948" alt="1cfa892e530fedf526ba3b805c326939" src="https://github.com/user-attachments/assets/e38dc4f2-520d-4ea0-8cce-ffebfe5b69d4" />

QLoRA fine-tuning for **Qwen3** plus a **RAG** stack (FAISS + embeddings) for **ECE269-style** linear-algebra / homework Q&A. Includes **Streamlit UI**, CLI chat, optional **RAG evaluation** on JSONL benchmarks, and PDF chunking utilities.

## Fine-tuned weights (QLoRA adapter)

**QLoRA checkpoint weights are not included in this repository** (file size limits). To obtain the fine-tuned adapter (e.g. `checkpoint-2400`), please email **zhc085@ucsd.edu**. Alternatively, run `Qlora_Finetune/train_openr1_math.py` to train your own.

## Features

- **Fine-tuning**: `Qlora_Finetune/train_openr1_math.py` — QLoRA on OpenR1-style math data with Hugging Face `Trainer`.
- **RAG**: `rag_pipeline/chat_rag.py` — retrieve Top-K chunks with **BGE** + **FAISS**, then generate with **Qwen3** + optional **PEFT** adapter; **4-bit** loading supported when CUDA is available.
- **UI**: `rag_pipeline/chat_rag_ui.py` — Streamlit sidebar (model path, `rag_root`, Top-K, temperature, etc.).
- **Data**: `rag_pipeline/data_preparation/` — `chunk.py` (PDF → JSONL, edit paths in `main()`), `build_vector_db.py` (JSONL → `ece269.index` / `ece269_meta.json` under `rag_pipeline/data/vector_db/`).
- **Eval**: `rag_pipeline/evaluation/` — `run_rag_eval.py` + `sample_benchmark.jsonl` (retrieval / answer metrics).

## Repository layout

```
Qlora-RAG-math-chatbot/
├── Qlora_Finetune/
│   └── train_openr1_math.py      # QLoRA training entry
├── Qlora-result/                 # Adapter checkpoints (e.g. checkpoint-2400), gitignored if large
├── llm_scoring/
│   └── chat_openr1_compare.py    # Interactive base vs fine-tuned compare (CLI)
├── rag_pipeline/
│   ├── __init__.py
│   ├── chat_rag.py               # RAG CLI + load_generator / load_retriever / search
│   ├── chat_rag_ui.py            # Streamlit app
│   ├── data/
│   │   ├── chunks/               # *.jsonl chunks (input to build_vector_db)
│   │   ├── vector_db/            # ece269.index, ece269_meta.json (FAISS + metadata)
│   │   └── ECE269doc/            # Optional: course PDFs (your layout may vary)
│   ├── data_preparation/
│   │   ├── chunk.py              # HW PDF pairing → jsonl (configure paths inside main())
│   │   └── build_vector_db.py    # Embeds chunks → writes vector_db/
│   ├── retrieval/
│   │   └── retriever.py          # Small standalone retriever example (paths may need edits)
│   └── evaluation/
│       ├── run_rag_eval.py       # Batch eval from JSONL
│       ├── metrics.py
│       └── sample_benchmark.jsonl
├── requirements.txt
├── pyproject.toml
├── README.md
└── README.zh.md
```

## Requirements

- Python **≥ 3.10**
- **PyTorch**: install **before** the rest — use [pytorch.org](https://pytorch.org) for a **CUDA** build if you use a GPU; CPU-only wheels are very slow for 4B-class models.
- Other deps: see `requirements.txt` (Transformers, PEFT, bitsandbytes, sentence-transformers, faiss-cpu, streamlit, pymupdf, …).

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -U pip
# Then install torch (CUDA example — pick your CUDA from pytorch.org):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quick start

### 1) Build / refresh the vector index

From the **repository root**, after `rag_pipeline/data/chunks/*.jsonl` exist:

```bash
python rag_pipeline/data_preparation/build_vector_db.py
```

Outputs: `rag_pipeline/data/vector_db/ece269.index` and `ece269_meta.json`.

### 2) RAG chat (CLI)

```bash
python rag_pipeline/chat_rag.py --load_in_4bit
```

Defaults: `rag_root` = `rag_pipeline/data` (indexes under `vector_db/` or `RAG/`). If `Qlora-result/checkpoint-2400` exists, it is auto-used as adapter. Flags: `--adapter_path`, `--top_k`, `--show_sources`, etc. (`python rag_pipeline/chat_rag.py -h`).

### 3) RAG chat (Streamlit)

```bash
streamlit run rag_pipeline/chat_rag_ui.py
```

Run from repo root so `rag_pipeline` imports resolve. Sidebar: set **Adapter Path** to a folder that contains `adapter_config.json` (e.g. `Qlora-result/checkpoint-2400`).

### 4) QLoRA training

```bash
python Qlora_Finetune/train_openr1_math.py
```

Adjust dataset / `output_dir` / model id inside the script or extend CLI as needed.

### 5) RAG benchmark (optional)

```bash
python -m rag_pipeline.evaluation.run_rag_eval --benchmark rag_pipeline/evaluation/sample_benchmark.jsonl --output rag_pipeline/evaluation/last_report.json --load_in_4bit
```

### 6) Base vs fine-tuned compare (optional)

```bash
python llm_scoring/chat_openr1_compare.py --adapter_path Qlora-result/checkpoint-2400 --load_in_4bit
```

## Configuration notes

| Item | Description |
|------|-------------|
| **Index location** | `chat_rag.resolve_index_meta` searches `{rag_root}/vector_db/` then `{rag_root}/RAG/` for `{variant}.index` and `{variant}_meta.json`, and also falls back to `rag_pipeline/data`. |
| **GPU** | If `torch.cuda.is_available()` is false, 4-bit is skipped and the LLM loads on CPU (slow). Install a CUDA-enabled PyTorch build. |
| **chunk.py** | No argparse; edit `hw_pdf` / `sol_pdf` / `out_path` in `main()` for your PDFs. |

## License / course use

Use course PDFs only in line with your institution’s copyright and collaboration policies.

## See also

- **Chinese README**: [README.zh.md](README.zh.md)
