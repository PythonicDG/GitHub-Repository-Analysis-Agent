from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import logging

from src.github_loader import GitHubRepoLoader
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore
from utils import setup_logging

# Initialize FastAPI
app = FastAPI(title="GitHub Analysis Agent API")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class IngestRequest(BaseModel):
    repo_name: Optional[str] = None
    topic: Optional[str] = None

class ChatRequest(BaseModel):
    question: str

class RepoMetadata(BaseModel):
    name: str
    description: Optional[str]
    stars: int
    forks: int
    language: Optional[str]

# Global state (for simplicity in this demo)
# In a real app, you might track per-session vector stores or multiple repos
current_repo_meta = None

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/ingest")
async def ingest_repo(request: IngestRequest):
    global current_repo_meta
    try:
        loader = GitHubRepoLoader()
        if request.repo_name:
            content = loader.load_repo(request.repo_name)
            contents = [content]
        elif request.topic:
            contents = loader.search_repos(request.topic, max_results=1) # Just one for speed in UI
            if not contents:
                raise HTTPException(status_code=404, detail="No repositories found for this topic")
        else:
            raise HTTPException(status_code=400, detail="Provide repo_name or topic")

        # Process and Store
        processor = DocumentProcessor()
        documents = processor.process_many(contents)
        
        store = ChromaVectorStore()
        store.add_documents(documents)

        # Save metadata for UI
        repo = contents[0]
        current_repo_meta = {
            "name": repo.repo_name,
            "description": repo.metadata.description,
            "stars": repo.metadata.stars,
            "forks": repo.metadata.forks,
            "language": repo.metadata.language,
            "contributors": repo.metadata.contributors[:5],
            "tree_entries": len(repo.directory_tree.splitlines())
        }

        return {"status": "success", "repo_metadata": current_repo_meta}

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        pipeline = RAGPipeline()
        answer = pipeline.ask(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {"current_repo": current_repo_meta}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
