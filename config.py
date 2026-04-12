"""
config.py
------------------------------------------------------------
Flat application settings loaded from .env file.
Uses pydantic-settings for type-safe, validated config.
------------------------------------------------------------
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

    # Logging level
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = Settings()
