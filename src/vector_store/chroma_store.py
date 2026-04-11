"""
src/vector_store/chroma_store.py
----------------------------------------------------------
Manages the ChromaDB vector store — handles persisting
document embeddings to disk and similarity-based retrieval.

ChromaDB is used as a lightweight, local vector database
that requires zero infrastructure.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from src.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Wraps ChromaDB behind a simple interface for adding
    documents and performing similarity search.
    """

    COLLECTION_NAME = "github_repos"

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        self._embedding_manager = embedding_manager or EmbeddingManager()
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._store: Optional[Chroma] = None

    @property
    def store(self) -> Chroma:
        """Lazily create or load the Chroma vector store."""
        if self._store is None:
            logger.info("Initializing ChromaDB at: %s", self._persist_dir)
            self._store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self._embedding_manager.embeddings,
                persist_directory=self._persist_dir,
            )
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add.")
            return

        logger.info("Adding %d documents to the vector store…", len(documents))
        self.store.add_documents(documents)
        logger.info("Documents added successfully.")

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-k most similar documents for a query."""
        logger.debug("Similarity search: query='%s', k=%d", query, k)
        return self.store.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Return a LangChain retriever backed by this store."""
        return self.store.as_retriever(
            search_kwargs=search_kwargs or {"k": 5},
        )
