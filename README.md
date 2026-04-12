# 🤖 GitAnalyzer AI — Simplified

A lean, efficient GitHub repository analysis agent that focuses on provide core insights with minimal complexity and token usage.

---

## ✨ Features

- **Direct Repo Analysis** — Input any GitHub URL or `owner/repo` to start.
- **Efficient Extraction** — Fetches metadata, file tree, and key project files (package.json, Dockerfile, etc.).
- **Hybrid Chat Logic** — Instantly answers factual questions (stars, structure) via rule-based logic to save tokens.
- **LLM-Powered Insights** — Uses Groq (Llama-3.1-8b) with selective context for complex reasoning.
- **Zero-DB Architecture** — No vector store or embeddings required; uses simple JSON caching for speed.

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
