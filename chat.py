"""
chat.py
------------------------------------------------------------
Hybrid chat module — uses rule-based logic for simple queries
and the Groq LLM for complex reasoning.

Design:
  1. Check if the question matches a "rule" (structure, stats,
     dependencies) → answer instantly without any LLM call.
  2. If not, build a small, selective context from the cached
     repo data and send it to the LLM.
  3. Context budget is kept tight (~2000 tokens) to minimize
     Groq API usage.
------------------------------------------------------------
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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
# Rule-based answers (no LLM needed)
# ------------------------------------------------------------------

def _try_rule_based(question: str, repo_data: dict) -> str | None:
    """
    Attempt to answer the question without calling the LLM.

    Returns the answer string if a rule matches, or None if
    the LLM should handle it.
    """
    q = question.lower().strip()
    meta = repo_data.get("metadata", {})

    # --- File structure / tree ---
    if any(kw in q for kw in ["file structure", "directory", "folder", "tree", "file list", "project structure"]):
        tree = repo_data.get("file_tree", [])
        if not tree:
            return "No file tree information is available for this repository."

        # Build a readable tree
        dirs = [e["path"] for e in tree if e["type"] == "dir"]
        files = [e["path"] for e in tree if e["type"] == "file"]

        lines = [f"📁 **{repo_data['name']}** — File Structure\n"]
        lines.append(f"Total: {len(files)} files, {len(dirs)} directories\n")

        # Show top-level entries first
        top_level = sorted(set(
            p.split("/")[0] for p in [e["path"] for e in tree]
        ))
        lines.append("**Top-level:**")
        for item in top_level[:30]:
            is_dir = any(e["path"] == item and e["type"] == "dir" for e in tree) or \
                     any(e["path"].startswith(item + "/") for e in tree)
            icon = "📂" if is_dir else "📄"
            lines.append(f"  {icon} {item}")

        if len(top_level) > 30:
            lines.append(f"  ... and {len(top_level) - 30} more")

        return "\n".join(lines)

    # --- Stars / forks / stats ---
    if any(kw in q for kw in ["how many stars", "star count", "stars"]) and "star" in q:
        return f"⭐ **{repo_data['name']}** has **{meta.get('stars', 'N/A'):,}** stars."

    if any(kw in q for kw in ["how many forks", "fork count", "forks"]) and "fork" in q:
        return f"🍴 **{repo_data['name']}** has **{meta.get('forks', 'N/A'):,}** forks."

    if any(kw in q for kw in ["what language", "programming language", "written in", "tech stack"]):
        lang = meta.get("language", "Unknown")
        topics = meta.get("topics", [])
        answer = f"💻 Primary language: **{lang}**"
        if topics:
            answer += f"\n🏷️ Topics: {', '.join(topics)}"
        return answer

    # --- Dependencies ---
    if any(kw in q for kw in ["dependencies", "packages", "requirements", "what does it use"]):
        key_files = repo_data.get("key_files", {})
        dep_files = {k: v for k, v in key_files.items()
                     if k in ("requirements.txt", "package.json", "pyproject.toml", "Cargo.toml", "go.mod")}

        if not dep_files:
            return "No dependency files (requirements.txt, package.json, etc.) were found in this repository."

        lines = [f"📦 **Dependencies for {repo_data['name']}:**\n"]
        for fname, content in dep_files.items():
            lines.append(f"**{fname}:**")
            lines.append(f"```\n{content}\n```\n")
        return "\n".join(lines)

    # --- License ---
    if "license" in q:
        return f"📄 License: **{meta.get('license', 'Unknown')}**"

    # --- Contributors ---
    if any(kw in q for kw in ["contributor", "who made", "who built", "author"]):
        contributors = meta.get("contributors", [])
        if contributors:
            return f"👥 Top contributors: {', '.join(contributors)}"
        return "Contributor information is not available."

    # --- Basic "what is this repo" ---
    if any(kw in q for kw in ["what is this", "describe", "overview", "about this repo", "summary"]):
        desc = meta.get("description", "No description")
        lang = meta.get("language", "Unknown")
        stars = meta.get("stars", 0)
        return (
            f"## {repo_data['name']}\n\n"
            f"{desc}\n\n"
            f"- 💻 Language: **{lang}**\n"
            f"- ⭐ Stars: **{stars:,}**\n"
            f"- 🍴 Forks: **{meta.get('forks', 0):,}**\n"
            f"- 📄 License: **{meta.get('license', 'Unknown')}**\n"
            f"- 🏷️ Topics: {', '.join(meta.get('topics', [])) or 'None'}"
        )

    # No rule matched — let the LLM handle it
    return None


# ------------------------------------------------------------------
# LLM-based answer
# ------------------------------------------------------------------

def _build_context(repo_data: dict, question: str) -> str:
    """
    Build a selective context string for the LLM.

    Keeps it compact to minimize token usage:
      - Always include metadata summary
      - Include README (truncated)
      - Include relevant key files based on the question
      - Include abbreviated file tree
    """
    parts = []

    # 1. Metadata summary (always included, ~100 tokens)
    meta = repo_data.get("metadata", {})
    parts.append(
        f"[REPOSITORY INFO]\n"
        f"Name: {repo_data['name']}\n"
        f"URL: {repo_data['url']}\n"
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

    # 4. Key files — include ones that seem relevant to the question
    key_files = repo_data.get("key_files", {})
    q_lower = question.lower()

    for fname, content in key_files.items():
        # Always include config files (small and useful)
        # Or include if the filename is mentioned in the question
        fname_lower = fname.lower()
        if (fname_lower in q_lower or
            any(kw in q_lower for kw in ["config", "setup", "docker", "ci", "deploy", "build"]) or
            fname in ("requirements.txt", "package.json", "pyproject.toml")):
            # Limit each file to 1500 chars in context
            parts.append(f"[FILE: {fname}]\n{content[:1500]}")

    return "\n\n---\n\n".join(parts)


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

    Uses a hybrid approach:
      1. Try rule-based answer first (free, instant)
      2. Fall back to LLM with selective context

    Returns: {"answer": str, "source": "rule" | "llm"}
    """
    if not repo_data:
        return {
            "answer": "No repository is loaded. Please add a repository first.",
            "source": "rule",
        }

    # Try rule-based answer
    rule_answer = _try_rule_based(question, repo_data)
    if rule_answer:
        logger.info("Answered with rule-based logic (no LLM call)")
        return {"answer": rule_answer, "source": "rule"}

    # Fall back to LLM
    logger.info("Using LLM for question: %s", question[:80])
    context = _build_context(repo_data, question)
    llm_answer = _ask_llm(question, context)
    return {"answer": llm_answer, "source": "llm"}
