"""
main.py
============================================================
Entry point for the AI-Powered GitHub Repository Analysis Agent.

Usage:
    python main.py                          # Interactive mode
    python main.py --repo owner/repo        # Analyze a specific repo
    python main.py --topic "machine learning" # Search by topic

The agent will:
  1. Fetch the repository/repositories from GitHub
  2. Process and chunk the content (README, docs, source)
  3. Store embeddings in ChromaDB
  4. Enter an interactive Q&A loop powered by RAG
============================================================
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import settings
from src.agent import build_agent_graph
from src.document_processor import DocumentProcessor
from src.github_loader import GitHubRepoLoader
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore
from utils import setup_logging

console = Console()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# CLI argument parser
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered GitHub Repository Analysis Agent",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--repo",
        type=str,
        help="Full repository name to analyze (e.g. 'langchain-ai/langchain')",
    )
    group.add_argument(
        "--topic",
        type=str,
        help="Topic to search for repositories (e.g. 'machine learning')",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------

def ingest(repo_name: str | None = None, topic: str | None = None) -> None:
    """Fetch, process, and store repository content."""

    # Step 1 — Load
    loader = GitHubRepoLoader()
    if repo_name:
        console.print(f"\n[bold cyan]📦 Loading repository:[/] {repo_name}")
        contents = [loader.load_repo(repo_name)]
    elif topic:
        console.print(f"\n[bold cyan]🔍 Searching repositories for topic:[/] {topic}")
        contents = loader.search_repos(topic, max_results=3)
    else:
        console.print("[red]Error: provide --repo or --topic[/]")
        sys.exit(1)

    console.print(f"[green]✓ Loaded {len(contents)} repository(ies)[/]")

    # Step 2 — Process
    console.print("[bold cyan]📝 Processing documents…[/]")
    processor = DocumentProcessor()
    documents = processor.process_many(contents)
    console.print(f"[green]✓ Created {len(documents)} document chunks[/]")

    # Step 3 — Store
    console.print("[bold cyan]💾 Storing embeddings in ChromaDB…[/]")
    store = ChromaVectorStore()
    store.add_documents(documents)
    console.print("[green]✓ Embeddings stored successfully[/]\n")


def interactive_qa() -> None:
    """Run an interactive Q&A loop using the RAG pipeline."""
    console.print(
        Panel(
            "[bold]Ask any question about the ingested repository.\n"
            "Type [cyan]exit[/cyan] or [cyan]quit[/cyan] to stop.[/]",
            title="💬 Interactive Q&A",
            border_style="bright_blue",
        )
    )

    pipeline = RAGPipeline()

    while True:
        question = Prompt.ask("\n[bold yellow]Your question")
        if question.lower().strip() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye! 👋[/]")
            break

        with console.status("[cyan]Thinking…[/]"):
            answer = pipeline.ask(question)

        console.print(Panel(answer, title="🤖 Answer", border_style="green"))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    setup_logging()

    console.print(
        Panel(
            "[bold bright_white]AI-Powered GitHub Repository Analysis Agent[/]\n"
            "[dim]Fetch · Process · Embed · Ask — powered by RAG[/]",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )

    args = parse_args()

    # If no arguments, prompt the user interactively
    if not args.repo and not args.topic:
        choice = Prompt.ask(
            "[bold]Enter a GitHub repo name or a search topic",
            default="langchain-ai/langchain",
        )
        # Simple heuristic: if it contains '/', treat as repo name
        if "/" in choice:
            args.repo = choice
        else:
            args.topic = choice

    # Run the ingestion pipeline
    ingest(repo_name=args.repo, topic=args.topic)

    # Enter interactive Q&A
    interactive_qa()


if __name__ == "__main__":
    main()
