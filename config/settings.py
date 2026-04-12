"""
config/settings.py
----------------------------------------------------------
Application settings loaded from environment variables.
Uses pydantic-settings for type-safe, validated config.
----------------------------------------------------------
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- GitHub ---
    github_token: str = ""

    # --- Groq ---
    groq_api_key: str = ""

    # --- Gemini ---
    google_api_key: str = ""

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma_db"

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- LLM ---
    llm_provider: str = "groq"  # "groq" or "google"
    llm_model: str = "llama-3.1-8b-instant" 
    # For Groq: llama-3.3-70b-versatile
    # For Gemini: gemini-2.0-flash, gemini-1.5-pro

    # --- Document Processing ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Logging ---
    log_level: str = "INFO"


# Singleton settings instance used throughout the project
settings = Settings()
