"""
src/embeddings/embedding_manager.py
----------------------------------------------------------
Manages the embedding model used to convert text chunks
into dense vector representations.

Uses sentence-transformers via LangChain's
HuggingFaceEmbeddings wrapper for local, free embeddings.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Provides a configured LangChain-compatible embedding model.

    Default model: 'all-MiniLM-L6-v2' (fast, lightweight, good quality).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initialize and return the embedding model."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully.")
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""
        return self.embeddings.embed_documents(texts)
