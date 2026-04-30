"""
chat.py
RAG chat module — retrieves context via ChromaDB semantic
search and falls back to JSON-based context if needed.
"""

import logging
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config import settings

logger = logging.getLogger(__name__)

# System prompt
SYSTEM_PROMPT = """\
You are a GitHub repository analysis assistant. Answer questions
about the repository using ONLY the context provided below.

RULES:
1. Use ONLY the provided context. Do NOT use prior knowledge.
2. If the context doesn't have enough info, say so clearly.
3. Never fabricate file names, functions, or statistics.
4. Cite file paths when referencing specific files.
5. Be concise but thorough.
6. If the provided code snippets are truncated, state exactly which part is missing instead of guessing the implementation.

REPOSITORY CONTEXT:
{context}
"""

# Keywords that indicate the user needs raw source code
_CODE_KEYWORDS = [
    "code", "implement", "fix", "error", "how to",
    "write", "change", "logic for", "function", "class",
    "method", "import", "variable", "debug", "snippet", "example",
]


# RAG context builder

def _build_context_from_rag(retrieved: dict, question: str) -> str:
    """
    Build context string from ChromaDB retrieval results.

    This function aggregates different types of retrieved documents (metadata, 
    README, file tree, summaries, and source code) into a single context string 
    formatted for the LLM. It maintains a specific section order and respects 
    token/character limits.

    Preserves the same section format as the original JSON builder:
      [REPOSITORY INFO], [README], [FILE TREE],
      [REPOSITORY INDEX], [DETAILED FILE DESCRIPTIONS],
      [RAW SOURCE CODE]
    """
    parts: list[str] = []
    q_lower = question.lower()

    # 1. Metadata (always included, ~100 tokens)
    metadata_docs = retrieved.get("metadata_docs", [])
    for doc in metadata_docs:
        parts.append(f"[REPOSITORY INFO]\n{doc.get('content', '')}")

    # 2. README (always included, trimmed to budget)
    readme_docs = retrieved.get("readme_docs", [])
    if readme_docs:
        readme_text = "\n".join(doc.get("content", "") for doc in readme_docs)
        parts.append(f"[README]\n{readme_text[:2500]}")

    # 3. File tree (always included)
    tree_docs = retrieved.get("tree_docs", [])
    if tree_docs:
        tree_text = tree_docs[0].get("content", "")
        parts.append(f"[FILE TREE — top-level entries]\n{tree_text}")

    # 4. Repository index (1-line per file from ALL summaries)
    all_summaries = retrieved.get("all_summaries", [])
    if all_summaries:
        index_lines: list[str] = []
        # Sort by path for consistent ordering
        sorted_summaries = sorted(
            all_summaries, 
            key=lambda x: x.get("metadata", {}).get("path", "")
        )
        
        for s in sorted_summaries:
            metadata = s.get("metadata", {})
            path = metadata.get("path", "unknown")
            content = s.get("content", "")
            
            # Skip the filepath prefix line to get the actual summary
            lines = content.split("\n")
            summary_line = ""
            for line in lines:
                line = line.strip()
                if line and line != path:
                    summary_line = line[:100]
                    break
            if summary_line:
                index_lines.append(f"- {path}: {summary_line}")
        
        if index_lines:
            parts.append(
                "[REPOSITORY INDEX]\n" + "\n".join(index_lines[:40])
            )

    # 5. Detailed file descriptions (from semantic search results)
    relevant = retrieved.get("relevant_docs", [])
    detailed_summaries = [
        r for r in relevant
        if r.get("metadata", {}).get("type") == "file_summary"
    ]
    
    if detailed_summaries:
        detail_parts: list[str] = []
        for s in detailed_summaries[:5]:
            metadata = s.get("metadata", {})
            path = metadata.get("path", "unknown")
            content = s.get("content", "")
            # Strip the filepath prefix from the content
            body = content.split("\n", 1)[1].strip() if "\n" in content else content
            detail_parts.append(f"### TECHNICAL MAP: {path}\n{body}")
        
        parts.append(
            "[DETAILED FILE DESCRIPTIONS]\n" + "\n\n".join(detail_parts)
        )

    # 6. Tree detail from semantic search (directory listings)
    tree_details = [
        r for r in relevant
        if r.get("metadata", {}).get("type") == "file_tree_detail"
    ]
    if tree_details:
        for td in tree_details[:2]:
            parts.append(f"[DIRECTORY DETAIL]\n{td.get('content', '')}")

    # 7. Raw source code (only when question implies need for code)
    needs_code = any(kw in q_lower for kw in _CODE_KEYWORDS)
    if needs_code:
        code_docs = [
            r for r in relevant
            if r.get("metadata", {}).get("type") == "file_content"
        ]
        if code_docs:
            code_parts: list[str] = []
            seen_paths: set[str] = set()
            for c in code_docs:
                metadata = c.get("metadata", {})
                path = metadata.get("path", "unknown")
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                content = c.get("content", "")
                # Strip the filepath prefix
                body = content.split("\n", 1)[1].strip() if "\n" in content else content
                code_parts.append(f"[FILE CONTENT: {path}]\n{body[:1500]}")
                if len(code_parts) >= 3:
                    break
            
            if code_parts:
                parts.append(
                    "[RAW SOURCE CODE]\n" + "\n\n".join(code_parts)
                )

    # Enforce ~5k token budget (~4 chars/token → 20,000 chars)
    full_context = "\n\n---\n\n".join(parts)
    return full_context[:20000]


# JSON fallback context builder

def _build_context(repo_data: dict, question: str) -> str:
    """
    Build a selective context string for the LLM.
    Uses the in-memory repo_data dict directly (no vector DB).
    """
    parts = []
    q_lower = question.lower()

    # 1. Metadata summary (always included, ~100 tokens)
    meta = repo_data.get("metadata", {})
    parts.append(
        f"[REPOSITORY INFO]\n"
        f"Name: {repo_data.get('name', 'Unknown')}\n"
        f"URL: {repo_data.get('url', 'N/A')}\n"
        f"Description: {meta.get('description', 'N/A')}\n"
        f"Language: {meta.get('language', 'N/A')}\n"
        f"Stars: {meta.get('stars', 0)} | Forks: {meta.get('forks', 0)}\n"
        f"License: {meta.get('license', 'N/A')}\n"
        f"Topics: {', '.join(meta.get('topics', []))}\n"
        f"Contributors: {', '.join(meta.get('contributors', []))}\n"
        f"Open Issues: {meta.get('open_issues', 0)}\n"
        f"Default Branch: {meta.get('default_branch', 'main')}\n"
        f"Last Updated: {meta.get('last_updated', 'Unknown')}"
    )

    # 2. README (always included, already truncated during fetch)
    readme = repo_data.get("readme", "")
    if readme:
        # Further trim for LLM context if very long
        trimmed = readme[:2500]
        parts.append(f"[README]\n{trimmed}")

    # 3. File tree — abbreviated top-level listing
    tree = repo_data.get("file_tree", [])
    if tree:
        top_entries = sorted(set(
            e["path"].split("/")[0] for e in tree
        ))[:40]
        parts.append(f"[FILE TREE — top-level entries]\n" + "\n".join(top_entries))

    # 4. Codebase Map — Be selective to save tokens!
    summaries = repo_data.get("file_summaries", {})
    if summaries:
        relevant_summaries = []
        short_index = []

        for path, text in summaries.items():
            path_lower = path.lower()
            filename_lower = os.path.basename(path).lower()
            # If the filename or part of the path is mentioned in the question
            if filename_lower in q_lower or (path_lower in q_lower and len(path_lower) > 3):
                relevant_summaries.append(f"### TECHNICAL MAP: {path}\n{text}")
            else:
                # Otherwise, 1-line index
                first_line = text.split('\n')[0] if '\n' in text else text[:80]
                short_index.append(f"- {path}: {first_line}")

        if short_index:
            parts.append("[REPOSITORY INDEX]\n" + "\n".join(short_index[:40]))
        if relevant_summaries:
            # Limit to top 5 detailed maps to stay under 6k token limit
            parts.append("[DETAILED FILE DESCRIPTIONS]\n" + "\n\n".join(relevant_summaries[:5]))

    # 5. Key files — ONLY include raw code if implementation details are asked
    key_files = repo_data.get("key_files", {})
    needs_code = any(kw in q_lower for kw in _CODE_KEYWORDS)

    if needs_code:
        code_snippets = []
        for fname, content in key_files.items():
            fname_lower = fname.lower()
            basename_lower = os.path.basename(fname).lower()
            if basename_lower in q_lower or (fname_lower in q_lower and len(fname_lower) > 3):
                # Limit raw code to 1500 chars to respect Groq limits
                code_snippets.append(f"[FILE CONTENT: {fname}]\n{content[:1500]}")

        if code_snippets:
            # Only include top 3 files' raw code at once
            parts.append("[RAW SOURCE CODE]\n" + "\n\n".join(code_snippets[:3]))

    # FINAL STEP: Strictly enforce the 6k token budget by truncating characters.
    # Note: We use a conservative 4 characters per token estimate (20,000 chars ≈ 5,000 tokens).
    full_context = "\n\n---\n\n".join(parts)
    return full_context[:20000]


# LLM call

def _ask_llm(question: str, context: str) -> str:
    """Send the question + context to Groq LLM and return the answer."""
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0,
        groq_api_key=settings.groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# Public API

def chat(question: str, repo_data: dict, session_id: str = None) -> dict:
    """
    Answer a user question about the loaded repository.

    If session_id is provided, uses ChromaDB RAG retrieval.
    Falls back to JSON-based context if RAG is unavailable.

    Returns: {"answer": str, "source": "llm"}
    """
    if not repo_data:
        return {
            "answer": "No repository is loaded. Please add a repository first.",
            "source": "llm",
        }

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "source": "llm",
        }

    # --- Build context: RAG primary, JSON fallback ---
    context = None

    if session_id and settings.use_vector_db:
        try:
            import vector_store
            retrieved = vector_store.retrieve_context(session_id, question)
            # Verify we got meaningful results (at least metadata)
            if retrieved and retrieved.get("metadata_docs"):
                context = _build_context_from_rag(retrieved, question)
                logger.info(
                    "RAG context built: %d chars (session %s)",
                    len(context), session_id[:8],
                )
            else:
                logger.warning(
                    "RAG returned empty results for session %s, falling back to JSON",
                    session_id[:8],
                )
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed, falling back to JSON context: %s", exc
            )

    # Fallback to JSON-based context
    if context is None:
        context = _build_context(repo_data, question)
        logger.info("Using JSON fallback context: %d chars", len(context))

    # --- Send to LLM ---
    try:
        llm_answer = _ask_llm(question, context)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return {
            "answer": f"Sorry, the AI service returned an error: {str(e)}",
            "source": "llm",
        }

    return {"answer": llm_answer, "source": "llm"}
