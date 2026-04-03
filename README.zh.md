# ECE285 项目：QLoRA 微调与 RAG 系统 (含 Streamlit UI)

## 项目概览

本项目实现了一个结合 QLoRA 微调大语言模型（LLM）的检索增强生成（RAG）系统。核心思想是通过从知识库中检索相关信息来增强 LLM 的问答能力。项目提供了一个 Streamlit 网页界面，用于交互式 RAG 聊天。

本项目包含：
-   使用 QLoRA 微调 Qwen3 模型，以提高其在特定领域（如数学推理）的性能。
-   基于 LLM 的评分/评估组件，用于比较模型输出。
-   完整的 RAG 流水线，包括文档分块、向量数据库创建和检索逻辑。
-   一个用户友好的 Streamlit 网页 UI，用于 RAG 聊天机器人。

## 项目结构

仓库组织如下：

```
ECE285-Project/
├── qlora_finetune/
│   └── qwen3/
│       ├── train_openr1_math.py       # 数学推理的 QLoRA 微调脚本
│       └── <其他 Qwen3 微调相关脚本/模型>
├── llm_scoring/
│   └── chat_openr1_compare.py         # 用于评估/比较 LLM 结果的脚本
│   └── <其他 LLM 评分脚本>
├── rag_pipeline/
│   ├── data_preparation/
│   │   ├── chunk.py                   # 文档分块脚本
│   │   └── build_vector_db.py         # 向量数据库创建脚本
│   ├── retrieval/
│   │   └── retriever.py               # 检索逻辑脚本
│   ├── chat/
│   │   ├── chat_rag.py                # 核心 RAG 聊天逻辑 (命令行界面)
│   │   └── chat_rag_ui.py             # RAG 聊天的 Streamlit 网页 UI
│   
├── data/
│   ├── raw_docs/                      # 存放原始文档 (例如：PDF, TXT 文件)
│   │   └── ECE269rag/                 # ECE269 课程相关原始文档
│   │       ├── Midtern Solution.pdf
│   │       ├── Quiz 1 Solutions.pdf
│   │       └── ...
│   ├── chunks/                        # 存放处理后的文本块 (.jsonl 文件)
│   │   ├── ece269_concepts_cunks.jsonl
│   │   ├── calculus_concepts_chunks.jsonl
│   │   └── ...
│   └── vector_db/                     # 存放向量数据库索引和元数据
│       ├── ece269.index
│       ├── ece269_meta.json
│       ├── calculus.index
│       └── calculus_meta.json
├── main_scripts/                      # 项目主入口或编排脚本
│   └── <例如：orchestrator.py>
├── skills/                            # 技能模块 (例如：linear-algebra-solver)
│   └── linear-algebra-solver/
│       └── ...
├── .gitignore
├── README.md
├── README.zh.md
├── pyproject.toml
└── uv.lock
```

## 设置

1.  **克隆仓库：**
    ```bash
    git clone <repository_url>
    cd ECE285-Project
    ```
    (将 `<repository_url>` 替换为您的实际仓库 URL)

2.  **安装依赖：**
    建议使用虚拟环境。
    ```bash
    # 使用 uv (如果已安装)
    uv pip install -r requirements.txt

    # 或使用 pip
    pip install -r requirements.txt
    ```

## 使用说明

### 1. QLoRA 微调

要对 Qwen3 模型进行数学推理微调：

```bash
python qlora_finetune/qwen3/train_openr1_math.py
```
*(请确保您已在脚本中配置好所需数据集或通过参数传入。)*

### 2. LLM 评分/评估

运行基于 LLM 的评估：

```bash
python llm_scoring/chat_openr1_compare.py
```
*(此脚本通常提供一个交互式聊天，用于根据评分机制比较模型输出。)*

### 3. RAG 数据准备

在运行 RAG 聊天之前，您需要准备数据并构建向量数据库。

#### a. 文档分块
将原始文档处理成更小、更易于管理的块：
```bash
python rag_pipeline/data_preparation/chunk.py --input_dir data/raw_docs/ECE269rag --output_dir data/chunks --variant ece269
# 或用于微积分数据
python rag_pipeline/data_preparation/chunk.py --input_dir data/raw_docs/Calculus --output_dir data/chunks --variant calculus
```
*(根据您的原始文档和分块输出位置调整 `--input_dir` 和 `--output_dir`。)*

#### b. 构建向量数据库
从处理后的分块中创建 FAISS 索引和元数据：
```bash
python rag_pipeline/data_preparation/build_vector_db.py --chunk_dir data/chunks --output_dir data/vector_db --variant ece269
# 或用于微积分数据
python rag_pipeline/data_preparation/build_vector_db.py --chunk_dir data/chunks --output_dir data/vector_db --variant calculus
```
*(请确保 `--chunk_dir` 指向您的分块数据，`--output_dir` 是您期望的向量数据库位置。)*

### 4. RAG 聊天 (命令行界面)

通过命令行与 RAG 系统交互：

```bash
python rag_pipeline/chat/chat_rag.py --rag_variant ece269 --adapter_path qlora_finetune/qwen3/checkpoint-2400
# 您可以指定其他参数，例如 --load_in_4bit, --top_k 等。
```

### 5. RAG 聊天 (Streamlit 网页 UI)

对于交互式网页版 RAG 聊天机器人：

1.  **确保已安装 Streamlit**：`pip install streamlit` (如果尚未包含在 `requirements.txt` 中)。
2.  **运行 Streamlit 应用程序**：
    ```bash
    streamlit run rag_pipeline/chat/chat_rag_ui.py
    ```
    这将在您的网页浏览器中打开应用程序。您可以在侧边栏中配置各种参数，并与 RAG 机器人聊天。

## 配置

`chat_rag.py` 和 `chat_rag_ui.py` 的关键配置可以通过命令行参数（`chat_rag.py`）或 Streamlit 侧边栏（`chat_rag_ui.py`）进行调整：

*   `--base_model_name`：要使用的基础 LLM (例如，"Qwen/Qwen3-4B")。
*   `--adapter_path`：LoRA/QLoRA 适配器检查点路径。
*   `--load_in_4bit`：为基础模型使用 4 位量化。
*   `--rag_root`：RAG 数据的根目录 (默认值：`data`)。
*   `--rag_variant`：选择不同的 RAG 知识库 (例如，"ece269"，"calculus")。
*   `--embedding_model`：用于生成嵌入的模型 (例如，"BAAI/bge-base-en-v1.5")。
*   `--top_k`：检索到的最相关的块的数量。
*   `--system_prompt`：LLM 的自定义系统提示词。
