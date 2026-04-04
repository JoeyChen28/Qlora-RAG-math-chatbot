
# Qlora-RAG-math-chatbot
<img width="1912" height="948" alt="86505d47bde52818e0a297dfcd420238" src="https://github.com/user-attachments/assets/5f0bf93d-63f2-43e9-95b6-99b157967848" />


面向 **ECE269 线代 / 数学作业**场景的 **Qwen3 + QLoRA** 微调与 **RAG**（**FAISS** + 语义向量）问答项目，含 **Streamlit** 界面、命令行聊天、**JSONL 批量评测** 与 PDF 分块脚本。

## 微调权重（QLoRA 适配器）

因**文件体积限制**，本仓库**未包含 QLoRA 微调完成后的检查点/适配器参数**（如 `checkpoint-2400`）。如需获取，请发邮件至 **zhc085@![Uploading 86505d47bde52818e0a297dfcd420238.png…]()
ucsd.edu**。也可自行运行 `Qlora_Finetune/train_openr1_math.py` 复现训练。

## 功能概览

- **微调**：`Qlora_Finetune/train_openr1_math.py` — 基于 Hugging Face `Trainer` 的 QLoRA 训练。
- **RAG**：`rag_pipeline/chat_rag.py` — **BGE** 嵌入 + **FAISS** Top-K 检索，再用 **Qwen3** + 可选 **PEFT 适配器**生成；在 **CUDA 可用**时支持 **4bit** 加载。
- **界面**：`rag_pipeline/chat_rag_ui.py` — Streamlit 侧栏配置模型路径、`rag_root`、Top-K、温度等。
- **数据**：`rag_pipeline/data_preparation/chunk.py` 将作业 PDF 打成 JSONL（需在 `main()` 里改路径）；`build_vector_db.py` 将 `rag_pipeline/data/chunks/*.jsonl` 写入 `rag_pipeline/data/vector_db/ece269.index` 与 `ece269_meta.json`。
- **评测**：`rag_pipeline/evaluation/run_rag_eval.py` + `sample_benchmark.jsonl` — 检索与答案指标。

## 目录结构（与仓库一致）

```
Qlora-RAG-math-chatbot/
├── Qlora_Finetune/
│   └── train_openr1_math.py      # QLoRA 训练入口
├── Qlora-result/                 # 适配器输出目录（如 checkpoint-2400），大文件常 gitignore
├── llm_scoring/
│   └── chat_openr1_compare.py    # 基座 vs 微调 交互对比
├── rag_pipeline/
│   ├── chat_rag.py               # RAG 命令行核心逻辑
│   ├── chat_rag_ui.py            # Streamlit
│   ├── data/
│   │   ├── chunks/               # 分块 jsonl（建库输入）
│   │   ├── vector_db/            # ece269.index / ece269_meta.json
│   │   └── ECE269doc/            # 可选：课程 PDF（按你本地实际使用）
│   ├── data_preparation/
│   │   ├── chunk.py
│   │   └── build_vector_db.py
│   ├── retrieval/
│   │   └── retriever.py          # 简单检索示例（路径需自行对齐）
│   └── evaluation/
│       ├── run_rag_eval.py
│       ├── metrics.py
│       └── sample_benchmark.jsonl
├── requirements.txt
├── pyproject.toml
├── README.md
└── README.zh.md
```

## 环境依赖

- Python **≥ 3.10**
- **PyTorch**：须先安装；**有 NVIDIA 显卡请安装 CUDA 版**（见 [pytorch.org](https://pytorch.org)），否则大模型在 CPU 上会极慢。
- 其余：`pip install -r requirements.txt`（见文件内安装顺序说明）。

```bash
python -m venv venv
venv\Scripts\activate
pip install -U pip
# 先按官网安装 torch（GPU 示例）:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 使用说明

### 1）构建向量库

在仓库根目录，准备好 `rag_pipeline/data/chunks/` 下各 jsonl 后执行：

```bash
python rag_pipeline/data_preparation/build_vector_db.py
```

生成：`rag_pipeline/data/vector_db/ece269.index` 与 `ece269_meta.json`。

### 2）命令行 RAG

```bash
python rag_pipeline/chat_rag.py --load_in_4bit
```

默认 `rag_root` 为 `rag_pipeline/data`；若存在 `Qlora-result/checkpoint-2400` 会自动作为 adapter。更多参数：`python rag_pipeline/chat_rag.py -h`。

### 3）Streamlit

```bash
streamlit run rag_pipeline/chat_rag_ui.py
```

请在**仓库根目录**运行，保证能 `import rag_pipeline`。**Adapter Path** 需指向含 `adapter_config.json` 的目录（如 `Qlora-result/checkpoint-2400`）。

### 4）QLoRA 训练

```bash
python Qlora_Finetune/train_openr1_math.py
```

数据集路径、`output_dir` 等请在脚本内配置或自行扩展参数。

### 5）RAG 评测（可选）

```bash
python -m rag_pipeline.evaluation.run_rag_eval --benchmark rag_pipeline/evaluation/sample_benchmark.jsonl --output rag_pipeline/evaluation/last_report.json --load_in_4bit
```

### 6）模型对比（可选）

```bash
python llm_scoring/chat_openr1_compare.py --adapter_path Qlora-result/checkpoint-2400 --load_in_4bit
```

## 配置说明

| 项 | 说明 |
|----|------|
| **索引路径** | `resolve_index_meta` 会在 `{rag_root}/vector_db/`、`{rag_root}/RAG/` 下查找 `{variant}.index` 与元数据，并回退到 `rag_pipeline/data`。 |
| **GPU** | `torch.cuda.is_available()` 为 False 时会关闭 4bit 并在 CPU 上以 float32 加载，易表现为卡顿；请安装 CUDA 版 PyTorch。 |
| **chunk.py** | 无命令行参数，在 `main()` 中修改 PDF 与输出 jsonl 路径。 |

## 合规说明

课程 PDF、作业与解答的使用须遵守学校版权与学术诚信规定。

## 英文说明

见 [README.md](README.md)。
