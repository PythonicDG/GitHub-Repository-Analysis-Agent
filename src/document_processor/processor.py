"""
src/document_processor/processor.py
----------------------------------------------------------
Handles chunking and preparing raw repository content into
LangChain Document objects suitable for embedding and
vector storage.

Uses code-aware splitting for source files — splits on
class, function, and module boundaries rather than naive
character counts. Falls back to RecursiveCharacterTextSplitter
for prose (README, docs) and unsupported languages.

Also creates dedicated documents for repository metadata
and directory structure to enable structural queries.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import settings
from src.github_loader.loader import RepoContent

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Map file extensions → LangChain Language enum for code-aware splitting
# -------------------------------------------------------------------
EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".r": Language.PYTHON,      # R has no dedicated splitter — Python is closest
    ".jl": Language.PYTHON,     # Julia has no dedicated splitter — Python is closest
}


class DocumentProcessor:
    """
    Converts raw RepoContent into chunked LangChain Documents
    with rich metadata for downstream retrieval.

    Uses language-aware splitting for source code (splits on
    class / function / method boundaries) and standard text
    splitting for prose content.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Default text splitter for prose (README, docs)
        self._text_splitter = RecursiveCharacterTextSplitter(
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

        Creates documents for:
          - Repository metadata summary (source_type="metadata")
          - Directory tree (source_type="directory_tree")
          - README chunks (source_type="readme")
          - Documentation chunks (source_type="documentation")
          - Source code chunks (source_type="source_code")
        """
        documents: list[Document] = []
        base_meta = {
            "repo_name": repo_content.repo_name,
            "repo_url": repo_content.repo_url,
        }

        # 1. Repository metadata — single document for grounding
        metadata_summary = repo_content.metadata.to_summary()
        documents.append(Document(
            page_content=(
                f"Repository: {repo_content.repo_name}\n"
                f"URL: {repo_content.repo_url}\n"
                f"{metadata_summary}"
            ),
            metadata={**base_meta, "source_type": "metadata"},
        ))

        # 2. Directory tree — single document for structural queries
        if repo_content.directory_tree:
            documents.append(Document(
                page_content=(
                    f"Directory structure of {repo_content.repo_name}:\n\n"
                    f"{repo_content.directory_tree}"
                ),
                metadata={**base_meta, "source_type": "directory_tree"},
            ))

        # 3. README
        if repo_content.readme:
            docs = self._split_text(
                repo_content.readme,
                metadata={**base_meta, "source_type": "readme"},
            )
            documents.extend(docs)

        # 4. Documentation files
        for idx, doc_text in enumerate(repo_content.docs):
            docs = self._split_text(
                doc_text,
                metadata={**base_meta, "source_type": "documentation", "doc_index": idx},
            )
            documents.extend(docs)

        # 5. Source code files — code-aware chunking
        for file_info in repo_content.source_files:
            file_path = file_info["path"]
            docs = self._split_code(
                file_info["content"],
                file_path=file_path,
                metadata={
                    **base_meta,
                    "source_type": "source_code",
                    "file_path": file_path,
                },
            )
            documents.extend(docs)

        logger.info(
            "Processed repo '%s' → %d chunks "
            "(metadata=1, tree=%s, readme=%s, docs=%d, source=%d)",
            repo_content.repo_name,
            len(documents),
            bool(repo_content.directory_tree),
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

    def _get_code_splitter(self, file_path: str) -> RecursiveCharacterTextSplitter:
        """
        Return a language-aware splitter for the given file extension.

        Falls back to the generic text splitter if the language is
        not supported by LangChain's Language enum.
        """
        ext = PurePosixPath(file_path).suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)

        if language is not None:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

        # Unsupported language — fall back to generic splitter
        return self._text_splitter

    def _split_code(self, text: str, file_path: str, metadata: dict) -> list[Document]:
        """Split source code using a language-aware splitter."""
        splitter = self._get_code_splitter(file_path)
        return splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

    def _split_text(self, text: str, metadata: dict) -> list[Document]:
        """Split a text string into Documents with the given metadata."""
        return self._text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )
