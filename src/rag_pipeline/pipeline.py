"""
src/rag_pipeline/pipeline.py
----------------------------------------------------------
Implements the core Retrieval-Augmented Generation pipeline.

Flow:
  1. User asks a question about a repository.
  2. The retriever fetches the most relevant document chunks.
  3. The LLM generates an answer grounded in those chunks.

Uses LangChain's LCEL (LangChain Expression Language) for
a clean, composable chain definition.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import settings
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# System prompt template for the RAG chain
# -------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """\
You are an expert software engineer assistant. Use the following pieces of
context retrieved from a GitHub repository to answer the user's question.

If you don't know the answer based on the context, say so honestly — do NOT
fabricate information.

Context:
{context}

Question: {question}

Provide a clear, concise, and technically accurate answer:
"""


class RAGPipeline:
    """
    Orchestrates retrieval + generation for repository Q&A.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self._vector_store = vector_store or ChromaVectorStore()
        self._llm_model = llm_model or settings.llm_model

        # Build LLM using Groq (free, fast inference)
        self._llm = ChatGroq(
            model=self._llm_model,
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )

        # Build RAG chain using LCEL
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """
        Ask a question and receive an answer grounded in
        the repository content stored in the vector database.
        """
        logger.info("RAG query: %s", question)
        answer = self._chain.invoke({"question": question})
        return answer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chain(self):
        """Construct the LCEL retrieval-augmented generation chain."""
        from operator import itemgetter

        retriever = self._vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Chain: retrieve context → format prompt → LLM → parse string
        chain = (
            {
                "context": itemgetter("question") | retriever | self._format_docs,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )
        return chain

    @staticmethod
    def _format_docs(docs) -> str:
        """Combine retrieved documents into a single context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
