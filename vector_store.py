"""
vector_store.py
ChromaDB-backed vector store for the RAG pipeline.
Handles:
  - Chunking and embedding repo data
  - Semantic retrieval
  - Collection lifecycle management
"""

import hashlib
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import settings

logger = logging.getLogger(__name__)

# Singletons

_chroma_client = None
_embedding_fn = None


def _get_client() -> chromadb.ClientAPI:
    """Lazy-init persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialized at %s", persist_dir)
    return _chroma_client


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    """Lazy-init sentence-transformer embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        logger.info("Embedding model loaded: %s", settings.embedding_model)
    return _embedding_fn


def _collection_name(session_id: str) -> str:
    """
    Derive a valid ChromaDB collection name from a session ID.
    ChromaDB names: 3-63 chars, alphanumeric start/end.
    """
    hashed = hashlib.md5(session_id.encode()).hexdigest()[:16]
    return f"repo_{hashed}"


# Text chunking

def _chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[str]:
    """
    Split text into overlapping chunks, breaking at natural
    boundaries (newlines) when possible.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Prefer breaking at a newline in the back half of the chunk
        if end < len(text):
            last_nl = chunk.rfind("\n", int(chunk_size * 0.5))
            if last_nl != -1:
                end = start + last_nl + 1
                chunk = text[start:end]

        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)

        # Advance; overlap keeps continuity between chunks
        start = end - overlap if end < len(text) else len(text)

    return chunks


# Ingestion

def ingest_repo_data(session_id: str, repo_data: dict) -> int:
    """
    Chunk and embed all repo data into a session-scoped
    ChromaDB collection.

    Document types stored:
      metadata        — always-retrieved repo info
      readme          — README chunks
      file_tree       — top-level directory listing
      file_tree_detail— per-directory file listings
      file_summary    — LLM-generated summaries per file
      file_content    — raw source code chunks

    Returns the number of documents ingested.
    """
    client = _get_client()
    col_name = _collection_name(session_id)

    # Fresh collection for each ingestion (handles re-analysis)
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=col_name,
        embedding_function=_get_embedding_fn(),
        # Use cosine similarity as it performs well for high-dimensional text embeddings
        metadata={"hnsw:space": "cosine"},
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    doc_id = 0

    def _add(text: str, meta: dict):
        nonlocal doc_id
        if text and text.strip():
            documents.append(text.strip())
            metadatas.append(meta)
            ids.append(f"doc_{doc_id}")
            doc_id += 1

    # ---- 1. Repository metadata (always retrieved) ----
    meta = repo_data.get("metadata", {})
    meta_text = (
        f"Repository: {repo_data.get('name', 'Unknown')}\n"
        f"URL: {repo_data.get('url', 'N/A')}\n"
        f"Description: {meta.get('description', 'N/A')}\n"
        f"Primary Language: {meta.get('language', 'N/A')}\n"
        f"Stars: {meta.get('stars', 0)} | Forks: {meta.get('forks', 0)}\n"
        f"License: {meta.get('license', 'N/A')}\n"
        f"Topics: {', '.join(meta.get('topics', []))}\n"
        f"Contributors: {', '.join(meta.get('contributors', []))}\n"
        f"Open Issues: {meta.get('open_issues', 0)}\n"
        f"Default Branch: {meta.get('default_branch', 'main')}\n"
        f"Last Updated: {meta.get('last_updated', 'Unknown')}"
    )
    _add(meta_text, {"type": "metadata", "path": "_metadata"})

    # ---- 2. README (chunked for better retrieval) ----
    readme = repo_data.get("readme", "")
    if readme:
        for i, chunk in enumerate(_chunk_text(readme)):
            _add(chunk, {"type": "readme", "path": "README.md", "chunk_index": i})

    # ---- 3. File tree ----
    tree = repo_data.get("file_tree", [])
    if tree:
        # Top-level overview
        top_entries = sorted(set(
            e["path"].split("/")[0] for e in tree
        ))[:60]
        tree_text = (
            "Repository file structure (top-level entries):\n"
            + "\n".join(top_entries)
        )
        _add(tree_text, {"type": "file_tree", "path": "_file_tree"})

        # Per-directory detail for granular structure queries
        dir_groups: dict[str, list[str]] = {}
        for e in tree:
            parts = e["path"].split("/")
            top_dir = parts[0] if len(parts) > 1 else "_root"
            dir_groups.setdefault(top_dir, []).append(e["path"])

        for dir_name, paths in dir_groups.items():
            if len(paths) > 1:
                detail = (
                    f"Files under '{dir_name}/' directory:\n"
                    + "\n".join(paths[:50])
                )
                _add(detail, {"type": "file_tree_detail", "path": dir_name})

    # ---- 4. File summaries (LLM-generated during fetch) ----
    summaries = repo_data.get("file_summaries", {})
    for path, summary in summaries.items():
        if summary and summary.strip():
            # Include filepath in text so semantic search can match on it
            _add(
                f"{path}\n\n{summary}",
                {"type": "file_summary", "path": path},
            )

    # ---- 5. File contents (raw source code, chunked) ----
    key_files = repo_data.get("key_files", {})
    for path, content in key_files.items():
        if content and content.strip():
            for i, chunk in enumerate(_chunk_text(content)):
                # Include filepath for retrieval relevance
                _add(
                    f"{path}\n\n{chunk}",
                    {"type": "file_content", "path": path, "chunk_index": i},
                )

    # ---- Batch insert into ChromaDB ----
    if documents:
        BATCH = 166  # ChromaDB recommended batch size
        for i in range(0, len(documents), BATCH):
            collection.add(
                documents=documents[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
                ids=ids[i : i + BATCH],
            )

    logger.info(
        "ChromaDB: ingested %d documents into '%s' for repo '%s'",
        len(documents),
        col_name,
        repo_data.get("name", "?"),
    )
    return len(documents)


# Retrieval

def retrieve_context(session_id: str, query: str, top_k: int = 12) -> dict:
    """
    Retrieve context for a query from the session's collection.

    Returns dict with:
      metadata_docs   – always-included repo info
      readme_docs     – always-included README chunks
      tree_docs       – always-included file tree
      all_summaries   – every file summary (for 1-line index)
      relevant_docs   – top-k semantically relevant results
    """
    client = _get_client()
    col_name = _collection_name(session_id)

    empty = {
        "metadata_docs": [],
        "readme_docs": [],
        "tree_docs": [],
        "all_summaries": [],
        "relevant_docs": [],
    }

    try:
        collection = client.get_collection(
            name=col_name,
            embedding_function=_get_embedding_fn(),
        )
    except Exception as exc:
        logger.error(
            "ChromaDB collection not found for session %s: %s",
            session_id[:8], exc,
        )
        return empty

    result = {k: list(v) for k, v in empty.items()}

    # 1. Always-included docs (fetched by type, no semantic ranking)
    for doc_type, key in [
        ("metadata", "metadata_docs"),
        ("readme", "readme_docs"),
        ("file_tree", "tree_docs"),
    ]:
        try:
            hits = collection.get(
                where={"type": doc_type},
                include=["documents", "metadatas"],
            )
            if hits and hits["documents"]:
                for doc, m in zip(hits["documents"], hits["metadatas"]):
                    result[key].append({"content": doc, "metadata": m})
        except Exception:
            pass

    # 2. All file summaries (for building the repository index)
    try:
        summary_hits = collection.get(
            where={"type": "file_summary"},
            include=["documents", "metadatas"],
        )
        if summary_hits and summary_hits["documents"]:
            for doc, m in zip(
                summary_hits["documents"], summary_hits["metadatas"]
            ):
                result["all_summaries"].append(
                    {"content": doc, "metadata": m}
                )
    except Exception:
        pass

    # 3. Semantic search across summaries, code, and tree details
    searchable_types = ["file_summary", "file_content", "file_tree_detail"]
    try:
        # Count searchable docs to avoid requesting more than exist
        searchable_ids = collection.get(
            where={"type": {"$in": searchable_types}},
            include=[],
        )
        searchable_count = (
            len(searchable_ids["ids"]) if searchable_ids else 0
        )

        if searchable_count > 0:
            n = min(top_k, searchable_count)
            search = collection.query(
                query_texts=[query],
                n_results=n,
                where={"type": {"$in": searchable_types}},
                include=["documents", "metadatas", "distances"],
            )

            if (
                search
                and search["documents"]
                and search["documents"][0]
            ):
                for doc, m, dist in zip(
                    search["documents"][0],
                    search["metadatas"][0],
                    search["distances"][0],
                ):
                    result["relevant_docs"].append(
                        {"content": doc, "metadata": m, "distance": dist}
                    )
    except Exception as exc:
        logger.warning("ChromaDB semantic search failed: %s", exc)

    return result


# Cleanup

def delete_collection(session_id: str) -> None:
    """Delete the ChromaDB collection for a session."""
    try:
        client = _get_client()
        col_name = _collection_name(session_id)
        client.delete_collection(col_name)
        logger.info("Deleted ChromaDB collection: %s", col_name)
    except Exception:
        pass
