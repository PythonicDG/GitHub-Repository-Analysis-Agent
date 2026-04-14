"""
github_fetcher.py
------------------------------------------------------------
Fetches repository data from the GitHub API.

Extracts only what's needed for analysis:
  - Basic metadata (stars, forks, language, etc.)
  - File tree (limited depth & count)
  - README content (truncated)
  - Key config files (package.json, requirements.txt, etc.)

Results are cached as JSON so repeated analysis is free.
------------------------------------------------------------
"""

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from github import Github, GithubException
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import settings

logger = logging.getLogger(__name__)

# Initialize LLM for summarization
_summarizer_llm = None

def get_summarizer():
    global _summarizer_llm
    if _summarizer_llm is None and settings.groq_api_key:
        _summarizer_llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    return _summarizer_llm

# Directory to cache fetched repo data
CACHE_DIR = Path("data/cache")

# Files we always try to fetch (important config / project files)
KEY_FILES = [
    "README.md", "readme.md", "README.rst",
    "package.json", "requirements.txt", "pyproject.toml",
    "setup.py", "setup.cfg", "Cargo.toml", "go.mod",
    "index.js", "app.js", "server.js", "main.js", "index.ts", "app.ts",
    "app.py", "main.py", "manage.py", "run.py", "config.py",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", ".github/workflows/ci.yml", ".github/workflows/main.yml",
    "LICENSE", "CONTRIBUTING.md", "CHANGELOG.md",
]

# Max limits to avoid fetching huge repos
MAX_TREE_ENTRIES = 1000    # max files in tree listing
MAX_README_CHARS = 5000    # truncate README beyond this
MAX_FILE_CHARS = 3000      # truncate individual files

# Directories/extensions to ignore
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__", ".venv", "dist", "build", ".next", ".github"}
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".md", ".json", 
    ".txt", ".yml", ".yaml", ".sh", ".c", ".cpp", ".h", ".go", ".rs", ".java", 
    ".php", ".rb", ".sql", ".xml", ".properties", ".ini", ".env"
}

# Max files to summarize (to avoid Groq rate limits during ingestion)
MAX_FILES_TO_SUMMARIZE = 30


# ------------------------------------------------------------------
# URL normalization
# ------------------------------------------------------------------

def normalize_repo_name(repo_input: str) -> str:
    """
    Convert any GitHub reference to 'owner/repo' format.

    Accepts:
      - "owner/repo"
      - "https://github.com/owner/repo"
      - "https://github.com/owner/repo.git"
      - "git@github.com:owner/repo.git"
    """
    repo_input = repo_input.strip()

    # Full HTTPS URL
    if repo_input.startswith(("https://", "http://")):
        parsed = urlparse(repo_input)
        path = parsed.path.strip("/").removesuffix(".git")
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

    # SSH URL
    ssh_match = re.match(r"git@github\.com:(.+?)(?:\.git)?$", repo_input)
    if ssh_match:
        return ssh_match.group(1)

    # Already owner/repo
    return repo_input.removesuffix(".git")


# ------------------------------------------------------------------
# Main fetch function
# ------------------------------------------------------------------

def fetch_repo(repo_input: str) -> dict:
    """
    Fetch structured repository data from GitHub.

    Returns a dict with: name, url, metadata, file_tree, readme,
    key_files (dict of filename → content).

    Results are cached to data/cache/ so the same repo isn't
    re-fetched unnecessarily.
    """
    repo_name = normalize_repo_name(repo_input)
    logger.info("Fetching repository: %s", repo_name)

    # Check cache first
    cached = _load_cache(repo_name)
    if cached:
        logger.info("Loaded %s from cache", repo_name)
        return cached

    # Connect to GitHub
    client = Github(settings.github_token) if settings.github_token else Github()

    try:
        repo = client.get_repo(repo_name)
    except GithubException as e:
        if e.status == 404:
            raise ValueError(f"Repository '{repo_name}' not found. Check the name or URL.")
        elif e.status == 403:
            raise ValueError("GitHub API rate limit exceeded. Try again later or add a token.")
        raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}")

    # Build result dict
    result = {
        "name": repo.full_name,
        "url": repo.html_url,
        "metadata": _extract_metadata(repo),
        "file_tree": [],
        "readme": "",
        "key_files": {},
        "file_summaries": {},  # New: Store LLM-generated summaries
    }

    # Fetch README
    try:
        readme_file = repo.get_readme()
        readme_text = readme_file.decoded_content.decode("utf-8", errors="replace")
        result["readme"] = readme_text[:MAX_README_CHARS]
        if len(readme_text) > MAX_README_CHARS:
            result["readme"] += "\n\n... (truncated)"
    except Exception:
        logger.debug("No README found for %s", repo_name)

    # Fetch file tree + key files in one pass
    try:
        tree = repo.get_git_tree(sha=repo.default_branch, recursive=True).tree
        entries = []
        
        # Priority mapping for summarization:
        # 1. Core Source Code (py, js, ts, etc.)
        # 2. Key Config (Dockerfile, package.json)
        # 3. Documentation (md, txt)
        # 4. Tests
        blobs_to_summarize = []

        for item in tree[:MAX_TREE_ENTRIES]:
            entry_type = "dir" if item.type == "tree" else "file"
            entries.append({"path": item.path, "type": entry_type})

            path_parts = set(item.path.split("/"))
            extension = os.path.splitext(item.path)[1].lower()
            
            if path_parts.intersection(IGNORE_DIRS):
                continue
                
            filename = os.path.basename(item.path)
            is_text = extension in TEXT_EXTENSIONS or filename.lower() in [k.lower() for k in KEY_FILES]

            if item.type == "blob" and is_text:
                # Add to a pending list with a priority score
                priority = 5 # Default
                if extension in ('.py', '.js', '.ts', '.go', '.rs'): priority = 1
                if filename.lower() in [k.lower() for k in KEY_FILES]: priority = 2
                if filename.lower() in ["core.py", "main.py", "app.py"]: priority = 0 # Highest priority
                if "test" in item.path.lower(): priority = 10 # Low priority
                if "docs" in item.path.lower(): priority = 8  # Low priority

                blobs_to_summarize.append({
                    "path": item.path, 
                    "sha": item.sha, 
                    "priority": priority
                })

        # Sort by priority (lowest score first)
        blobs_to_summarize.sort(key=lambda x: x["priority"])

        files_summarized = 0
        for blob_info in blobs_to_summarize:
            try:
                blob = repo.get_git_blob(blob_info["sha"])
                content = base64.b64decode(blob.content).decode("utf-8", errors="replace")
                truncated_content = content[:MAX_FILE_CHARS]
                
                if not truncated_content.strip():
                    continue

                result["key_files"][blob_info["path"]] = truncated_content

                if files_summarized < MAX_FILES_TO_SUMMARIZE:
                    summary = _summarize_file(blob_info["path"], truncated_content)
                    if summary:
                        result["file_summaries"][blob_info["path"]] = summary
                        files_summarized += 1
                        time.sleep(0.5) # Rate limit protection

            except Exception as e:
                logger.debug("Could not process file %s: %s", blob_info["path"], e)

        result["file_tree"] = entries

        if len(tree) > MAX_TREE_ENTRIES:
            logger.info("Tree truncated: %d entries (max %d)", len(tree), MAX_TREE_ENTRIES)

    except Exception as exc:
        logger.warning("Could not fetch tree for %s: %s", repo_name, exc)

    # Save to cache
    _save_cache(repo_name, result)

    logger.info(
        "Fetched %s: %d tree entries, %d key files, README=%s",
        repo_name, len(result["file_tree"]),
        len(result["key_files"]), bool(result["readme"])
    )

    return result


# ------------------------------------------------------------------
# Metadata extraction
# ------------------------------------------------------------------

def _extract_metadata(repo) -> dict:
    """Pull essential metadata fields from the repo object."""
    meta = {
        "description": repo.description or "No description",
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "language": repo.language or "Unknown",
        "topics": repo.get_topics(),
        "open_issues": repo.open_issues_count,
        "license": repo.license.name if repo.license else "Unknown",
        "default_branch": repo.default_branch or "main",
        "last_updated": repo.updated_at.isoformat() if repo.updated_at else "Unknown",
    }

    # Top 5 contributors (avoid rate limit issues)
    try:
        contributors = repo.get_contributors()
        meta["contributors"] = [c.login for c in contributors[:5]]
    except Exception:
        meta["contributors"] = []

    return meta


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def _cache_path(repo_name: str) -> Path:
    """Get the cache file path for a repo."""
    safe_name = repo_name.replace("/", "_")
    return CACHE_DIR / f"{safe_name}.json"


def _load_cache(repo_name: str) -> dict | None:
    """Load cached repo data if it exists."""
    path = _cache_path(repo_name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(repo_name: str, data: dict) -> None:
    """Save repo data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(repo_name)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Cached repo data to %s", path)


# ------------------------------------------------------------------
# LLM Summarization
# ------------------------------------------------------------------

def _summarize_file(filepath: str, content: str) -> str | None:
    """Use the LLM to generate a 1-2 sentence summary of a file."""
    llm = get_summarizer()
    if not llm:
        return None

    if not content.strip():
        return "Empty file."

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Technical Architect. Analyze the provided code snippet and return a structured summary:\n"
            "1. GLOBAL PURPOSE: One sentence summary of the file's role.\n"
            "2. KEY COMPONENTS: List main Classes and Functions with a 1-sentence description for each.\n"
            "3. LOGIC FLOW: A brief technical note on how data is processed here.\n"
            "Be extremely concise but technically accurate. Stick to the facts."
        )),
        ("human", "File: {filepath}\n\nContent:\n{content}")
    ])

    try:
        chain = prompt | llm
        response = chain.invoke({"filepath": filepath, "content": content[:1500]})
        return response.content.strip()
    except Exception as e:
        logger.warning("Summarization failed for %s: %s", filepath, e)
        return None
