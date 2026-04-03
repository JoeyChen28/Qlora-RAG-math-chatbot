# ECE285 Project: QLoRA Fine-tuning and RAG System with Streamlit UI

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system integrated with a QLoRA fine-tuned Large Language Model (LLM). The core idea is to enhance the LLM's ability to answer questions by retrieving relevant information from a knowledge base. A Streamlit web interface is provided for interactive RAG chat.

The project involves:
-   Fine-tuning a Qwen3 model using QLoRA for improved performance on specific domains (e.g., math reasoning).
-   An LLM-based scoring/evaluation component to compare model outputs.
-   A complete RAG pipeline including document chunking, vector database creation, and retrieval logic.
-   A user-friendly Streamlit web UI for the RAG chatbot.

## Project Structure

The repository is organized as follows:

```
ECE285-Project/
├── qlora_finetune/
│   └── qwen3/
│       ├── train_openr1_math.py       # QLoRA fine-tuning script for math reasoning
│       └── <Other Qwen3 fine-tuning scripts/models>
├── llm_scoring/
│   └── chat_openr1_compare.py         # Script for evaluating/comparing LLM results
│   └── <Other LLM scoring scripts>
├── rag_pipeline/
│   ├── data_preparation/
│   │   ├── chunk.py                   # Document chunking script
│   │   └── build_vector_db.py         # Vector database creation script
│   ├── retrieval/
│   │   └── retriever.py               # Retrieval logic script
│   ├── chat/
│   │   ├── chat_rag.py                # Core RAG chat logic (CLI)
│   │   └── chat_rag_ui.py             # Streamlit Web UI for RAG chat
│   
├── data/
│   ├── raw_docs/                      # Stores raw documents (e.g., PDF, TXT files)
│   │   └── ECE269rag/                 # ECE269 course related raw documents
│   │       ├── Midtern Solution.pdf
│   │       ├── Quiz 1 Solutions.pdf
│   │       └── ...
│   ├── chunks/                        # Stores processed text chunks (.jsonl files)
│   │   ├── ece269_concepts_cunks.jsonl
│   │   ├── calculus_concepts_chunks.jsonl
│   │   └── ...
│   └── vector_db/                     # Stores vector database indices and metadata
│       ├── ece269.index
│       ├── ece269_meta.json
│       ├── calculus.index
│       └── calculus_meta.json
├── main_scripts/                      # Main entry points or orchestration scripts
│   └── <e.g., orchestrator.py>
├── skills/                            # Skill modules (e.g., linear-algebra-solver)
│   └── linear-algebra-solver/
│       └── ...
├── .gitignore
├── README.md
├── README.zh.md
├── pyproject.toml
└── uv.lock
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ECE285-Project
    ```
    (Replace `<repository_url>` with your actual repository URL)

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    # Using uv (if installed)
    uv pip install -r requirements.txt

    # Or using pip
    pip install -r requirements.txt
    ```

## Usage

### 1. QLoRA Fine-tuning

To fine-tune the Qwen3 model for math reasoning:

```bash
python qlora_finetune/qwen3/train_openr1_math.py
```
*(Ensure you have the necessary dataset configured within the script or passed as arguments.)*

### 2. LLM Scoring/Evaluation

To run the LLM-based evaluation:

```bash
python llm_scoring/chat_openr1_compare.py
```
*(This script typically provides an interactive chat to compare model outputs based on a scoring mechanism.)*

### 3. RAG Data Preparation

Before running the RAG chat, you need to prepare your data and build the vector database.

#### a. Document Chunking
Process your raw documents into smaller, manageable chunks:
```bash
python rag_pipeline/data_preparation/chunk.py --input_dir data/raw_docs/ECE269rag --output_dir data/chunks --variant ece269
# Or for calculus data
python rag_pipeline/data_preparation/chunk.py --input_dir data/raw_docs/Calculus --output_dir data/chunks --variant calculus
```
*(Adjust `--input_dir` and `--output_dir` as per your raw document and chunk output locations.)*

#### b. Build Vector Database
Create the FAISS index and metadata from your processed chunks:
```bash
python rag_pipeline/data_preparation/build_vector_db.py --chunk_dir data/chunks --output_dir data/vector_db --variant ece269
# Or for calculus data
python rag_pipeline/data_preparation/build_vector_db.py --chunk_dir data/chunks --output_dir data/vector_db --variant calculus
```
*(Ensure the `--chunk_dir` points to your chunked data and `--output_dir` is your desired vector database location.)*

### 4. RAG Chat (Command Line Interface)

To interact with the RAG system via the command line:

```bash
python rag_pipeline/chat/chat_rag.py --rag_variant ece269 --adapter_path qlora_finetune/qwen3/checkpoint-2400
# You can specify other arguments like --load_in_4bit, --top_k, etc.
```

### 5. RAG Chat (Streamlit Web UI)

For an interactive web-based RAG chatbot:

1.  **Ensure Streamlit is installed**: `pip install streamlit` (if not already part of `requirements.txt`).
2.  **Run the Streamlit application**:
    ```bash
    streamlit run rag_pipeline/chat/chat_rag_ui.py
    ```
    This will open the application in your web browser. You can configure various parameters in the sidebar and chat with the RAG bot.

## Configuration

Key configurations for `chat_rag.py` and `chat_rag_ui.py` can be adjusted via command-line arguments (for `chat_rag.py`) or the Streamlit sidebar (for `chat_rag_ui.py`):

*   `--base_model_name`: Base LLM to use (e.g., "Qwen/Qwen3-4B").
*   `--adapter_path`: Path to the LoRA/QLoRA adapter checkpoint.
*   `--load_in_4bit`: Use 4-bit quantization for the base model.
*   `--rag_root`: Root directory for RAG data (default: `data`).
*   `--rag_variant`: Selects between different RAG knowledge bases (e.g., "ece269", "calculus").
*   `--embedding_model`: Model used for generating embeddings (e.g., "BAAI/bge-base-en-v1.5").
*   `--top_k`: Number of top relevant chunks to retrieve.
*   `--system_prompt`: Custom system prompt for the LLM.
