"""
src/github_loader/loader.py
----------------------------------------------------------
Responsible for fetching GitHub repositories — either by
topic search or by direct repository URL.

Uses PyGithub to interact with the GitHub REST API and
extracts README content, documentation files, source code,
repository metadata, and directory structure for downstream
processing.
----------------------------------------------------------
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from github import Github, Repository

from config import settings

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data containers
# -------------------------------------------------------------------

@dataclass
class RepoMetadata:
    """Rich metadata about a GitHub repository."""

    stars: int = 0
    forks: int = 0
    language: str = "Unknown"
    description: str = ""
    topics: list[str] = field(default_factory=list)
    open_issues: int = 0
    last_updated: str = ""
    default_branch: str = "main"
    license_name: str = "Unknown"
    contributors: list[str] = field(default_factory=list)  # top contributor logins

    def to_summary(self) -> str:
        """Format metadata as a human-readable summary string."""
        topics_str = ", ".join(self.topics) if self.topics else "None"
        contributors_str = ", ".join(self.contributors) if self.contributors else "Unknown"
        return (
            f"Description: {self.description}\n"
            f"Primary Language: {self.language}\n"
            f"Stars: {self.stars:,} | Forks: {self.forks:,} | Open Issues: {self.open_issues:,}\n"
            f"Topics: {topics_str}\n"
            f"License: {self.license_name}\n"
            f"Default Branch: {self.default_branch}\n"
            f"Last Updated: {self.last_updated}\n"
            f"Top Contributors: {contributors_str}"
        )


@dataclass
class RepoContent:
    """Container for all content extracted from a repository."""

    repo_name: str
    repo_url: str
    readme: str = ""
    docs: list[str] = field(default_factory=list)
    source_files: list[dict] = field(default_factory=list)  # {"path": ..., "content": ...}
    metadata: RepoMetadata = field(default_factory=RepoMetadata)
    directory_tree: str = ""  # full directory tree as a formatted string


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

    def _extract_metadata(self, repo: Repository.Repository) -> RepoMetadata:
        """Extract rich metadata from the repository."""
        metadata = RepoMetadata(
            stars=repo.stargazers_count,
            forks=repo.forks_count,
            language=repo.language or "Unknown",
            description=repo.description or "No description provided",
            topics=repo.get_topics(),
            open_issues=repo.open_issues_count,
            last_updated=repo.updated_at.isoformat() if repo.updated_at else "Unknown",
            default_branch=repo.default_branch or "main",
            license_name=repo.license.name if repo.license else "Unknown",
        )

        # Fetch top contributors (limit to 10 to avoid rate limits)
        try:
            contributors = repo.get_contributors()
            metadata.contributors = [c.login for c in contributors[:10]]
        except Exception:
            logger.debug("Could not fetch contributors for %s", repo.full_name)

        return metadata

    def _build_directory_tree(self, tree_items) -> str:
        """
        Build a visual directory tree string from git tree items.

        Produces output like:
          ├── src/
          │   ├── main.py
          │   └── utils/
          │       └── helpers.py
          ├── README.md
          └── requirements.txt
        """
        # Build a nested dict structure from flat paths
        root: dict = {}
        for item in tree_items:
            parts = item.path.split("/")
            current = root
            for i, part in enumerate(parts):
                if i == len(parts) - 1 and item.type == "blob":
                    # It's a file — store as None
                    current[part] = None
                else:
                    # It's a directory — create nested dict
                    if part not in current or current[part] is None:
                        current[part] = {}
                    current = current[part]

        # Render the tree recursively
        lines: list[str] = []
        self._render_tree(root, lines, prefix="")
        return "\n".join(lines)

    def _render_tree(self, node: dict, lines: list[str], prefix: str) -> None:
        """Recursively render a directory tree into lines."""
        entries = sorted(node.keys(), key=lambda x: (node[x] is not None, x))  # dirs first
        for i, name in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child = node[name]

            if child is None:
                # File
                lines.append(f"{prefix}{connector}{name}")
            else:
                # Directory
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if is_last else "│   "
                self._render_tree(child, lines, prefix + extension)

    def _extract_content(self, repo: Repository.Repository) -> RepoContent:
        """Extract README, docs, source files, metadata, and directory tree."""
        content = RepoContent(repo_name=repo.full_name, repo_url=repo.html_url)

        # Metadata
        content.metadata = self._extract_metadata(repo)

        # README
        try:
            readme = repo.get_readme()
            content.readme = readme.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            logger.debug("No README found for %s", repo.full_name)

        # Walk the full repository tree
        try:
            tree = repo.get_git_tree(sha=repo.default_branch, recursive=True).tree

            # Build directory tree string
            content.directory_tree = self._build_directory_tree(tree)

            for item in tree:
                if item.type != "blob":
                    continue

                path_lower = item.path.lower()

                # Documentation files (markdown in doc-like directories)
                if any(part in self.DOC_PATTERNS for part in path_lower.split("/")) and path_lower.endswith(".md"):
                    blob = repo.get_git_blob(item.sha)
                    text = base64.b64decode(blob.content).decode("utf-8", errors="replace")
                    content.docs.append(text)

                # Source code files
                elif any(path_lower.endswith(ext) for ext in self.CODE_EXTENSIONS):
                    blob = repo.get_git_blob(item.sha)
                    text = base64.b64decode(blob.content).decode("utf-8", errors="replace")
                    content.source_files.append({"path": item.path, "content": text})

        except Exception as exc:
            logger.warning("Could not walk tree for %s — %s", repo.full_name, exc)

        return content
