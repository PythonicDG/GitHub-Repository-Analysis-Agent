"""
config.py
Flat application settings loaded from .env file.
Uses pydantic-settings for type-safe, validated config.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app settings in one place — no nested packages."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GitHub Personal Access Token
    github_token: str = ""

    # Groq API Key (free tier)
    groq_api_key: str = ""

    # LLM model to use on Groq
    llm_model: str = "llama-3.1-8b-instant"

    # ChromaDB persistence directory
    chroma_persist_dir: str = "./data/chroma_db"

    # Use Vector DB (ChromaDB) for RAG? If False, falls back to JSON cache.
    use_vector_db: bool = False

    # Embedding model (sentence-transformers)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Document chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Logging level
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = Settings()
