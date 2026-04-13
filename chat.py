"""
chat.py
------------------------------------------------------------
RAG chat module — builds selective context from cached repo
data and uses the Groq LLM for all question answering.

Design:
  1. Build a token-budgeted context from cached repo data
     (metadata, file summaries, raw code snippets).
  2. Send the question + context to Groq LLM.
  3. Context budget is kept under ~5000 tokens to respect
     Groq free-tier limits.
------------------------------------------------------------
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System prompt — strict anti-hallucination
# ------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a GitHub repository analysis assistant. Answer questions
about the repository using ONLY the context provided below.

RULES:
1. Use ONLY the provided context. Do NOT use prior knowledge.
2. If the context doesn't have enough info, say so clearly.
3. Never fabricate file names, functions, or statistics.
4. Cite file paths when referencing specific files.
5. Be concise but thorough.

REPOSITORY CONTEXT:
{context}
"""




# ------------------------------------------------------------------
# LLM-based answer
# ------------------------------------------------------------------

def _build_context(repo_data: dict, question: str) -> str:
    """
    Build a selective context string for the LLM.
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
        f"Contributors: {', '.join(meta.get('contributors', []))}"
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
    # Check for keywords that actually require reading the raw source code
    needs_code = any(kw in q_lower for kw in ["code", "implement", "fix", "error", "how to", "write", "change", "logic for"])

    if needs_code:
        code_snippets = []
        for fname, content in key_files.items():
            fname_lower = fname.lower()
            basename_lower = os.path.basename(fname).lower()
            if basename_lower in q_lower or (fname_lower in q_lower and len(fname_lower) > 3):
                # Limit raw code to 1500 chars (down from 2500) to respect Groq limits
                code_snippets.append(f"[FILE CONTENT: {fname}]\n{content[:1500]}")
        
        if code_snippets:
            # Only include top 3 files' raw code at once
            parts.append("[RAW SOURCE CODE]\n" + "\n\n".join(code_snippets[:3]))

    # FINAL STEP: Strictly enforce the 6k token budget by truncating characters 
    # (~4 characters = 1 token, so 20,000 chars is roughly 5k tokens)
    full_context = "\n\n---\n\n".join(parts)
    return full_context[:20000]


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


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def chat(question: str, repo_data: dict) -> dict:
    """
    Answer a user question about the loaded repository.

    Builds selective context from cached repo data and sends
    the question to the Groq LLM.

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

    logger.info("Using LLM for question: %s", question[:80])
    context = _build_context(repo_data, question)
    try:
        llm_answer = _ask_llm(question, context)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return {
            "answer": f"Sorry, the AI service returned an error: {str(e)}",
            "source": "llm",
        }
    return {"answer": llm_answer, "source": "llm"}
