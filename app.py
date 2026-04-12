import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Import our simplified modules
from config import settings
import github_fetcher
import chat

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Simplified GitHub Analysis Agent")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class IngestRequest(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    question: str

# Global state for current repo data
# In a real app, use a session-based cache or database
current_repo_data = None

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/ingest")
async def ingest_repo(request: IngestRequest):
    global current_repo_data
    try:
        # Use our simplified fetcher
        current_repo_data = github_fetcher.fetch_repo(request.repo_url)
        return {"status": "success", "repo_metadata": current_repo_data["metadata"], "repo_name": current_repo_data["name"]}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process repository")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global current_repo_data
    if not current_repo_data:
        raise HTTPException(status_code=400, detail="Please ingest a repository first")
    
    try:
        # Use our simplified chat logic
        result = chat.chat(request.question, current_repo_data)
        return result
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

@app.get("/status")
async def get_status():
    if current_repo_data:
        return {"current_repo": current_repo_data["name"], "metadata": current_repo_data["metadata"]}
    return {"current_repo": None}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
