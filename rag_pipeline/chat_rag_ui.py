import streamlit as st
from pathlib import Path
import sys

import torch

# 仓库根目录必须在 sys.path 里，才能 `import rag_pipeline`（本文件在 rag_pipeline/ 下）
_rag_dir = Path(__file__).resolve().parent
_project_root = _rag_dir.parent
sys.path.insert(0, str(_project_root))

from rag_pipeline.chat_rag import (
    answer_question_rag,
    build_rag_args,
    load_generator,
    load_retriever,
    resolve_index_meta,
)

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Math RAG Chatbot", layout="wide")
st.title("Math RAG Chatbot")

# --- Initialize session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    base_model_name = st.text_input("Base Model Name", value="Qwen/Qwen3-4B")
    # 须指向含 adapter_config.json 的 PEFT 目录（本仓库实际为 Qlora-result/checkpoint-*）
    adapter_path_default = str(_project_root / "Qlora-result" / "checkpoint-2400")
    adapter_path = st.text_input(
        "Adapter Path（留空则只用基座模型）",
        value=adapter_path_default,
        help="例如 …/Qlora-result/checkpoint-2400；若训练输出在子目录 qwen3-4b-qlora-openr1-math 下则改到该路径。",
    )
    load_in_4bit = st.checkbox("Load 4-bit Quantized Model", value=True)
    
    # 与 chat_rag 默认一致：rag_pipeline/data（其下为 vector_db/ 或 RAG/）
    rag_root_default = str(_rag_dir / "data")
    rag_root = st.text_input("RAG Data Root Directory", value=rag_root_default)
    rag_variant = st.selectbox("RAG Variant", options=["ece269", "calculus"], index=0)
    
    embedding_model_name = st.text_input("Embedding Model", value="BAAI/bge-base-en-v1.5")
    query_instruction = st.text_area("Query Instruction", value="Represent this sentence for searching relevant passages: ")
    
    top_k = st.slider("Retrieval Top K", min_value=1, max_value=10, value=3)
    max_new_tokens = st.slider("Max New Tokens", min_value=100, max_value=2048, value=512)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05)
    
    system_prompt_default = "You are a helpful and concise mathematical assistant. Answer the user\'s question using ONLY the retrieved context below.\\nRule 1: Be direct and concise. Do not repeat the same point.\\nRule 2: Do not output any proof-ending symbols.\\nRule 3: Do not output source labels like [Source 1] in the text.\\n\\nContext:\\n{context}"
    system_prompt = st.text_area("System Prompt", value=system_prompt_default, height=150)
    
    show_sources = st.checkbox("Show Retrieved Sources", value=True)
    
    # Placeholder for loading status
    loading_status = st.empty()


# --- Cache models and retriever to avoid reloading on each interaction ---
@st.cache_resource
def cached_load_models_and_retriever(
    base_model_name, adapter_path, load_in_4bit, embedding_model_name, rag_root, rag_variant, index_file, meta_file
):
    adapter_path = adapter_path or None
    dummy_args = build_rag_args(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        rag_root=rag_root,
        rag_variant=rag_variant,
        index_file=index_file,
        meta_file=meta_file,
        embedding_model=embedding_model_name,
    )
    index_file_path, meta_file_path = resolve_index_meta(dummy_args)
    embed_model, index, metadata = load_retriever(dummy_args, index_file_path, meta_file_path)
    model, tokenizer = load_generator(dummy_args)
    return embed_model, index, metadata, model, tokenizer, index_file_path, meta_file_path


# Use the cached function to load models and retriever
with st.spinner("Loading models and retriever..."):
    embed_model, index, metadata, model, tokenizer, index_file_path, meta_file_path = cached_load_models_and_retriever(
        base_model_name, adapter_path, load_in_4bit, embedding_model_name, rag_root, rag_variant, None, None
    )
st.success("Models and retriever loaded successfully!")
_llm_dev = next(model.parameters()).device
_emb_cuda = torch.cuda.is_available()
st.caption(
    f"设备：Embedding 在 `{'cuda' if _emb_cuda else 'cpu'}` · LLM 权重在 `{_llm_dev}`。"
    f" 若 LLM 在 cpu 且本机有 NVIDIA 显卡，请安装 **CUDA 版 PyTorch**（见 pytorch.org），并确认 `torch.cuda.is_available()` 为 True。"
)

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main chat input ---
if prompt := st.chat_input("Ask the RAG chatbot..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            answer, retrieved_chunks = answer_question_rag(
                question=prompt,
                embed_model=embed_model,
                index=index,
                metadata=metadata,
                model=model,
                tokenizer=tokenizer,
                base_model_name=base_model_name,
                adapter_path=adapter_path or None,
                load_in_4bit=load_in_4bit,
                rag_root=rag_root,
                rag_variant=rag_variant,
                index_file=None,
                meta_file=None,
                embedding_model_name=embedding_model_name,
                query_instruction=query_instruction,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                system_prompt=system_prompt,
                keep_think=False,
                show_sources=show_sources,
            )
            st.markdown(answer)

            if show_sources and retrieved_chunks:
                st.subheader("Retrieved Sources:")
                for i, item in enumerate(retrieved_chunks, start=1):
                    source_name = f"Source {i}: {item.get('assignment', 'Unknown')}"
                    if item.get('problem_id'):
                        source_name += f" Q{item['problem_id']}"
                    if item.get('sub_id') and item['sub_id'] != 'main':
                        source_name += f"({item['sub_id']})"
                    if item.get('chunk_type'):
                        source_name += f" [{item['chunk_type']}]"
                    
                    st.expander(source_name).markdown(item.get('text', ''))

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
