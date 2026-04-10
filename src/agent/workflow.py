"""
src/agent/workflow.py
----------------------------------------------------------
Defines the LangGraph-based agent workflow.

The graph orchestrates the full pipeline:
  1. Fetch repository content     (load)
  2. Process & chunk documents    (process)
  3. Store embeddings in ChromaDB (store)
  4. Answer user questions via RAG (answer)

This is a placeholder scaffold — each node delegates to
the corresponding module. Extend with conditional edges,
tool-calling, or human-in-the-loop as the project grows.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.document_processor import DocumentProcessor
from src.github_loader import GitHubRepoLoader
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Shared state schema for the agent graph
# -------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """State passed between nodes in the LangGraph workflow."""

    repo_name: str          # GitHub repo identifier (e.g. "owner/repo")
    topic: str              # Topic for search-based loading
    question: str           # User's question about the repository
    answer: str             # Generated answer from RAG
    status: str             # Current pipeline status message


# -------------------------------------------------------------------
# Node functions
# -------------------------------------------------------------------

def load_repository(state: AgentState) -> AgentState:
    """Node: Fetch repository content from GitHub."""
    loader = GitHubRepoLoader()

    if state.get("repo_name"):
        content = loader.load_repo(state["repo_name"])
        contents = [content]
    elif state.get("topic"):
        contents = loader.search_repos(state["topic"], max_results=3)
    else:
        return {**state, "status": "Error: no repo_name or topic provided"}

    # Store raw contents in state for the next node
    state["_repo_contents"] = contents  # type: ignore[typeddict-unknown-key]
    return {**state, "status": f"Loaded {len(contents)} repo(s)"}


def process_documents(state: AgentState) -> AgentState:
    """Node: Chunk and prepare documents for embedding."""
    processor = DocumentProcessor()
    contents = state.get("_repo_contents", [])  # type: ignore[typeddict-item]
    documents = processor.process_many(contents)

    state["_documents"] = documents  # type: ignore[typeddict-unknown-key]
    return {**state, "status": f"Processed {len(documents)} document chunks"}


def store_embeddings(state: AgentState) -> AgentState:
    """Node: Embed and store document chunks in ChromaDB."""
    store = ChromaVectorStore()
    documents = state.get("_documents", [])  # type: ignore[typeddict-item]
    store.add_documents(documents)

    return {**state, "status": "Embeddings stored in ChromaDB"}


def answer_question(state: AgentState) -> AgentState:
    """Node: Answer the user's question using RAG."""
    question = state.get("question", "")
    if not question:
        return {**state, "answer": "", "status": "No question provided"}

    pipeline = RAGPipeline()
    answer = pipeline.ask(question)
    return {**state, "answer": answer, "status": "Answer generated"}


# -------------------------------------------------------------------
# Graph builder
# -------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """
    Construct and return the LangGraph agent workflow.

    Graph topology:
      load → process → store → answer → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("load", load_repository)
    graph.add_node("process", process_documents)
    graph.add_node("store", store_embeddings)
    graph.add_node("answer", answer_question)

    # Define edges (linear pipeline for now)
    graph.set_entry_point("load")
    graph.add_edge("load", "process")
    graph.add_edge("process", "store")
    graph.add_edge("store", "answer")
    graph.add_edge("answer", END)

    return graph
