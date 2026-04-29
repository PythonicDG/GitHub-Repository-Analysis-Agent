# 🤖 GitAnalyzer AI — Simplified

An intelligent agent that fetches GitHub repositories, processes their content, and lets you ask questions about them using a hybrid rule-based and LLM approach.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/import?repo=https://github.com/PythonicDG/GitHub-Repository-Analysis-Agent)

---

## ✨ Features

- **Direct Repo Analysis** — Analyze any public GitHub repository by providing the URL or `owner/repo`.
- **Efficient Data Extraction** — Automatically extracts metadata, file structures, and critical project files like `package.json` or `Dockerfile`.
- **Hybrid Retrieval (RAG)** — Seamlessly switches between high-speed JSON context and deep semantic search via ChromaDB (RAG).
- **Intelligent Context Management** — Dynamically optimizes token usage by selecting the most relevant codebase segments.

---

## 📁 Project Structure

```
├── app.py              # FastAPI backend & routes
├── github_fetcher.py   # Repo data extraction logic
├── chat.py             # Hybrid rule-based + LLM chat logic
├── config.py           # Flat application settings
├── requirements.txt    # Essential python dependencies
├── .env                # API keys & configuration
└── static/             # Premium dark-theme frontend
    ├── index.html
    ├── script.js
    └── style.css
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your keys:
```env
GITHUB_TOKEN=your_github_pat
GROQ_API_KEY=your_groq_key
LLM_MODEL=llama-3.1-8b-instant
LOG_LEVEL=INFO
```

### 3. Run the App

```bash
python app.py
```
Visit `http://localhost:8000` to start analyzing.

---

## 🔧 Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Backend         | FastAPI                             |
| LLM Provider    | Groq (Llama-3.1-8b-instant)         |
| Analysis        | Rule-based + Selective LLM Context  |
| GitHub API      | PyGithub                            |
| Frontend        | Vanilla HTML/JS/CSS (Premium Dark)  |

---

## 📄 License

MIT
