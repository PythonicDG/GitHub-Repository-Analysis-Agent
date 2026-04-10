# 🤖 AI-Powered GitHub Repository Analysis Agent

An intelligent agent that fetches GitHub repositories, processes their content
(README, documentation, source code), and lets you ask questions about them
using **Retrieval-Augmented Generation (RAG)**.

---

## ✨ Features

- **GitHub Integration** — Fetch repos by name or discover by topic
- **Smart Document Processing** — Chunks README, docs, and source code
- **Vector Storage** — Persists embeddings in ChromaDB for fast retrieval
- **RAG-Powered Q&A** — Ask natural-language questions about any repository
- **Agent Workflow** — Extensible LangGraph pipeline for orchestration

---

## 📁 Project Structure

```
├── main.py                         # Entry point — CLI + interactive Q&A
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Pydantic-settings configuration
│
├── src/
│   ├── github_loader/
│   │   └── loader.py               # Fetch repo content via GitHub API
│   ├── document_processor/
│   │   └── processor.py            # Chunk text into LangChain Documents
│   ├── embeddings/
│   │   └── embedding_manager.py    # sentence-transformers embeddings
│   ├── vector_store/
│   │   └── chroma_store.py         # ChromaDB vector database wrapper
│   ├── rag_pipeline/
│   │   └── pipeline.py             # Retrieval + generation chain
│   └── agent/
│       └── workflow.py             # LangGraph agent workflow
│
└── utils/
    └── helpers.py                  # Logging setup & shared utilities
```

---

## 🚀 Quick Start

### 1. Clone & Create Virtual Environment

```bash
git clone <your-repo-url>
cd "GitHub Repository Analysis Agent"
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env
# Edit .env and fill in your API keys
```

### 4. Run the Agent

```bash
# Analyze a specific repository
python main.py --repo langchain-ai/langchain

# Search by topic
python main.py --topic "machine learning"

# Interactive mode (will prompt you)
python main.py
```

---

## 🔧 Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.10+                        |
| LLM Framework  | LangChain + LCEL                    |
| LLM Provider   | Groq (free, fast inference)          |
| Agent Framework | LangGraph                           |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB       | ChromaDB                            |
| GitHub API      | PyGithub                            |
| Config          | pydantic-settings + .env            |
| CLI / UX        | Rich                                |

---

## 📋 Environment Variables

| Variable           | Description                        | Default              |
|--------------------|------------------------------------|----------------------|
| `GITHUB_TOKEN`     | GitHub Personal Access Token       | —                    |
| `GROQ_API_KEY`     | Groq API key (free at console.groq.com) | —               |
| `EMBEDDING_MODEL`  | sentence-transformers model name   | `all-MiniLM-L6-v2`  |
| `LLM_MODEL`        | LLM model identifier              | `llama-3.3-70b-versatile` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path            | `./data/chroma_db`  |
| `CHUNK_SIZE`       | Text chunk size (characters)       | `1000`               |
| `CHUNK_OVERLAP`    | Overlap between chunks             | `200`                |
| `LOG_LEVEL`        | Logging level                      | `INFO`               |

---

## 📄 License

MIT
