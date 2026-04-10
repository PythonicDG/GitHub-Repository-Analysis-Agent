"""
src/github_loader/loader.py
----------------------------------------------------------
Responsible for fetching GitHub repositories — either by
topic search or by direct repository URL.

Uses PyGithub to interact with the GitHub REST API and
extracts README content, documentation files, and source
code for downstream processing.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from github import Github, Repository

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RepoContent:
    """Container for the raw content extracted from a repository."""

    repo_name: str
    repo_url: str
    readme: str = ""
    docs: list[str] = field(default_factory=list)
    source_files: list[dict] = field(default_factory=list)  # {"path": ..., "content": ...}


class GitHubRepoLoader:
    """
    Fetches repository content from GitHub.

    Supports two modes:
      1. Load a single repo by its full name (e.g. "owner/repo")
         or full URL (e.g. "https://github.com/owner/repo").
      2. Search repositories by topic and load the top results.
    """

    # File extensions we consider as source code
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c",
        ".rb", ".php", ".swift", ".kt", ".scala", ".r", ".jl",
    }

    # Documentation file patterns
    DOC_PATTERNS = {"doc", "docs", "documentation", "wiki", "guide"}

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or settings.github_token
        self._client = Github(self._token) if self._token else Github()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_repo_name(repo_input: str) -> str:
        """
        Convert any GitHub reference to 'owner/repo' format.

        Accepts:
          - "owner/repo"
          - "https://github.com/owner/repo"
          - "https://github.com/owner/repo.git"
          - "git@github.com:owner/repo.git"
        """
        repo_input = repo_input.strip()

        # Handle full HTTPS URLs
        if repo_input.startswith(("https://", "http://")):
            parsed = urlparse(repo_input)
            # path is like "/owner/repo" or "/owner/repo.git"
            path = parsed.path.strip("/").removesuffix(".git")
            parts = path.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"

        # Handle SSH URLs (git@github.com:owner/repo.git)
        ssh_match = re.match(r"git@github\.com:(.+?)(?:\.git)?$", repo_input)
        if ssh_match:
            return ssh_match.group(1)

        # Already in owner/repo format — just strip trailing .git
        return repo_input.removesuffix(".git")

    def load_repo(self, repo_name: str) -> RepoContent:
        """Load content from a single repository by name or URL."""
        repo_name = self._normalize_repo_name(repo_name)
        logger.info("Loading repository: %s", repo_name)
        repo = self._client.get_repo(repo_name)
        return self._extract_content(repo)

    def search_repos(self, topic: str, max_results: int = 5) -> list[RepoContent]:
        """Search repositories by topic and return extracted content."""
        logger.info("Searching repositories for topic: '%s'", topic)
        results: list[RepoContent] = []

        for repo in self._client.search_repositories(query=f"topic:{topic}", sort="stars")[:max_results]:
            try:
                results.append(self._extract_content(repo))
            except Exception as exc:
                logger.warning("Skipping repo %s — %s", repo.full_name, exc)

        logger.info("Loaded %d repositories for topic '%s'", len(results), topic)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_content(self, repo: Repository.Repository) -> RepoContent:
        """Extract README, docs, and source files from a repository."""
        content = RepoContent(repo_name=repo.full_name, repo_url=repo.html_url)

        # README
        try:
            readme = repo.get_readme()
            content.readme = readme.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            logger.debug("No README found for %s", repo.full_name)

        # Walk repository tree (first level only to avoid rate-limit exhaustion)
        try:
            tree = repo.get_git_tree(sha=repo.default_branch, recursive=True).tree
            for item in tree:
                if item.type != "blob":
                    continue

                path_lower = item.path.lower()

                # Documentation files (markdown in doc-like directories)
                if any(part in self.DOC_PATTERNS for part in path_lower.split("/")) and path_lower.endswith(".md"):
                    blob = repo.get_git_blob(item.sha)
                    import base64
                    text = base64.b64decode(blob.content).decode("utf-8", errors="replace")
                    content.docs.append(text)

                # Source code files
                elif any(path_lower.endswith(ext) for ext in self.CODE_EXTENSIONS):
                    blob = repo.get_git_blob(item.sha)
                    import base64
                    text = base64.b64decode(blob.content).decode("utf-8", errors="replace")
                    content.source_files.append({"path": item.path, "content": text})

        except Exception as exc:
            logger.warning("Could not walk tree for %s — %s", repo.full_name, exc)

        return content
