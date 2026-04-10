"""
src/document_processor/processor.py
----------------------------------------------------------
Handles chunking and preparing raw repository content into
LangChain Document objects suitable for embedding and
vector storage.

Splits text using RecursiveCharacterTextSplitter with
configurable chunk size and overlap for optimal retrieval.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.github_loader.loader import RepoContent

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Converts raw RepoContent into chunked LangChain Documents
    with rich metadata for downstream retrieval.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, repo_content: RepoContent) -> list[Document]:
        """
        Process a single RepoContent into a flat list of Documents.

        Each document carries metadata:
          - repo_name, repo_url
          - source_type: "readme" | "documentation" | "source_code"
          - file_path (for source code files)
        """
        documents: list[Document] = []
        base_meta = {
            "repo_name": repo_content.repo_name,
            "repo_url": repo_content.repo_url,
        }

        # 1. README
        if repo_content.readme:
            docs = self._split_text(
                repo_content.readme,
                metadata={**base_meta, "source_type": "readme"},
            )
            documents.extend(docs)

        # 2. Documentation files
        for idx, doc_text in enumerate(repo_content.docs):
            docs = self._split_text(
                doc_text,
                metadata={**base_meta, "source_type": "documentation", "doc_index": idx},
            )
            documents.extend(docs)

        # 3. Source code files
        for file_info in repo_content.source_files:
            docs = self._split_text(
                file_info["content"],
                metadata={
                    **base_meta,
                    "source_type": "source_code",
                    "file_path": file_info["path"],
                },
            )
            documents.extend(docs)

        logger.info(
            "Processed repo '%s' → %d chunks (readme=%s, docs=%d, source=%d)",
            repo_content.repo_name,
            len(documents),
            bool(repo_content.readme),
            len(repo_content.docs),
            len(repo_content.source_files),
        )
        return documents

    def process_many(self, contents: list[RepoContent]) -> list[Document]:
        """Process multiple RepoContent objects into a single document list."""
        all_docs: list[Document] = []
        for content in contents:
            all_docs.extend(self.process(content))
        return all_docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str, metadata: dict) -> list[Document]:
        """Split a text string into Documents with the given metadata."""
        return self._splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
