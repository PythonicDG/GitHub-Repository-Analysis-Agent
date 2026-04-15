"""
app.py
FastAPI application for the GitHub Repository Analysis Agent.

Features:
- Session-based repo storage
- Non-blocking ingestion
- CORS middleware
- Automatic session cleanup
"""

import asyncio
import logging
import os
import uuid
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from config import settings
import github_fetcher
import chat

if settings.use_vector_db:
    import vector_store

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="GitHub Repository Analysis Agent")

# CORS middleware — allows frontend to call API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Session-based storage
# Each user gets a unique session_id stored in a cookie.
# Their repo data is stored in this dict, keyed by session_id.

sessions: dict[str, dict] = {}  # {session_id: {"repo_data": ..., "last_active": timestamp}}

SESSION_TIMEOUT = 3600  # 1 hour — auto-cleanup inactive sessions
MAX_SESSIONS = 50       # Safety cap to prevent memory exhaustion


def _get_session_id(request: Request) -> str | None:
    """Extract session ID from cookie."""
    return request.cookies.get("session_id")


def _create_session(response: Response) -> str:
    """Create a new session and set the cookie."""
    session_id = str(uuid.uuid4())
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=SESSION_TIMEOUT,
        samesite="lax",
    )
    sessions[session_id] = {"repo_data": None, "last_active": time.time()}
    logger.info("Created new session: %s (total: %d)", session_id[:8], len(sessions))
    return session_id


def _cleanup_sessions():
    """Remove expired sessions and their ChromaDB collections."""
    now = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if now - data.get("last_active", 0) > SESSION_TIMEOUT
    ]
    for sid in expired:
        if settings.use_vector_db:
            vector_store.delete_collection(sid)
        del sessions[sid]
        logger.info("Cleaned up expired session: %s", sid[:8])


def _ensure_session(request: Request, response: Response) -> str:
    """Get existing session or create a new one."""
    # Periodic cleanup
    if len(sessions) > MAX_SESSIONS // 2:
        _cleanup_sessions()

    session_id = _get_session_id(request)
    if session_id and session_id in sessions:
        sessions[session_id]["last_active"] = time.time()
        return session_id

    # Enforce max sessions
    if len(sessions) >= MAX_SESSIONS:
        _cleanup_sessions()
        if len(sessions) >= MAX_SESSIONS:
            # Force-remove oldest
            oldest = min(sessions, key=lambda s: sessions[s]["last_active"])
            if settings.use_vector_db:
                vector_store.delete_collection(oldest)
            del sessions[oldest]
            logger.warning("Force-removed oldest session to make room")

    return _create_session(response)


# Request/Response Models

class IngestRequest(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    question: str


# Routes

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/ingest")
async def ingest_repo(request_body: IngestRequest, request: Request, response: Response):
    session_id = _ensure_session(request, response)

    try:
        # Run the blocking fetch in a thread pool so it doesn't freeze
        # the server for other users
        repo_data = await asyncio.to_thread(
            github_fetcher.fetch_repo, request_body.repo_url
        )
        sessions[session_id]["repo_data"] = repo_data
        logger.info("Session %s: fetched %s", session_id[:8], repo_data["name"])

        # Embed into ChromaDB for RAG retrieval if enabled
        if settings.use_vector_db:
            doc_count = await asyncio.to_thread(
                vector_store.ingest_repo_data, session_id, repo_data
            )
            logger.info(
                "Session %s: embedded %d documents into ChromaDB",
                session_id[:8], doc_count,
            )

        return {
            "status": "success",
            "repo_metadata": repo_data["metadata"],
            "repo_name": repo_data["name"],
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Ingestion error (session %s): %s", session_id[:8], str(e))
        raise HTTPException(status_code=500, detail="Failed to process repository")


@app.post("/chat")
async def chat_endpoint(request_body: ChatRequest, request: Request, response: Response):
    session_id = _ensure_session(request, response)
    session = sessions.get(session_id, {})
    repo_data = session.get("repo_data")

    if not repo_data:
        raise HTTPException(status_code=400, detail="Please ingest a repository first")

    try:
        # Run the LLM call in a thread pool to stay non-blocking
        # Pass session_id so chat can use ChromaDB RAG retrieval
        result = await asyncio.to_thread(
            chat.chat, request_body.question, repo_data, session_id
        )
        return result
    except Exception as e:
        logger.error("Chat error (session %s): %s", session_id[:8], str(e))
        return {"answer": f"An error occurred: {str(e)}", "source": "error"}


@app.get("/status")
async def get_status(request: Request, response: Response):
    session_id = _ensure_session(request, response)
    session = sessions.get(session_id, {})
    repo_data = session.get("repo_data")

    if repo_data:
        return {
            "current_repo": repo_data["name"],
            "metadata": repo_data["metadata"],
            "active_sessions": len(sessions),
        }
    return {"current_repo": None, "active_sessions": len(sessions)}


# Entry point

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
