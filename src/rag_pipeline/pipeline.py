"""
src/rag_pipeline/pipeline.py
----------------------------------------------------------
Implements the core Retrieval-Augmented Generation pipeline.

Flow:
  1. User asks a question about a repository.
  2. The retriever fetches the most relevant document chunks.
  3. The LLM generates an answer grounded in those chunks.

Uses a strict system prompt that forces the model to answer
ONLY from the provided context and cite sources. This
significantly reduces hallucinations.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
from operator import itemgetter
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Strict system prompt — anti-hallucination design
# -------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a precise GitHub repository analysis assistant. You answer questions \
about software repositories using ONLY the context provided below.

## STRICT RULES — YOU MUST FOLLOW ALL OF THEM:

1. **ONLY use information from the provided context.** Do NOT use prior \
knowledge, training data, or general assumptions.
2. **If the context does not contain enough information to answer the \
question, you MUST say:** "I don't have enough information in the \
repository context to answer this question."
3. **Never fabricate** file names, function names, class names, URLs, \
statistics, or any other details not explicitly present in the context.
4. **Cite your sources.** When referencing code or files, mention the \
file path (e.g., `src/main.py`) if available in the metadata.
5. **Be precise about uncertainty.** If something is partially covered \
in the context, say what you know and explicitly state what is missing.
6. **Use the repository metadata** (stars, forks, language, contributors) \
when answering questions about the repository itself.

## CONTEXT FROM REPOSITORY:

{context}
"""

HUMAN_PROMPT = "{question}"


class RAGPipeline:
    """
    Orchestrates retrieval + generation for repository Q&A.

    Features:
      - Strict anti-hallucination system prompt
      - Source citations from document metadata
      - Repository metadata injected into context
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self._vector_store = vector_store or ChromaVectorStore()
        self._llm_model = llm_model or settings.llm_model

        # Select LLM Provider
        provider = settings.llm_provider.lower()
        
        if provider == "groq":
            logger.info(f"Using Groq LLM: {self._llm_model}")
            self._llm = ChatGroq(
                model=self._llm_model,
                temperature=0,
                groq_api_key=settings.groq_api_key,
            )
        elif provider == "google":
            # If the user forgot to change the model when switching to google,
            # we'll default to gemini to avoid validation errors.
            if "llama" in self._llm_model.lower():
                logger.warning(f"Provider 'google' detected with Llama model '{self._llm_model}'. Defaulting to 'gemini-2.0-flash'.")
                self._llm_model = "gemini-2.0-flash"

            logger.info(f"Using Google Gemini LLM: {self._llm_model}")
            self._llm = ChatGoogleGenerativeAI(
                model=self._llm_model,
                temperature=0,
                google_api_key=settings.google_api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

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
        retriever = self._vector_store.as_retriever(search_kwargs={"k": 8})

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

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
        """
        Format retrieved documents with their metadata for context.

        Each chunk is prefixed with its source type and file path
        so the LLM can cite specific files in its answer.
        """
        formatted_parts: list[str] = []

        for doc in docs:
            meta = doc.metadata
            source_type = meta.get("source_type", "unknown")
            file_path = meta.get("file_path", "")
            repo_name = meta.get("repo_name", "")

            # Build a header line for this chunk
            if source_type == "source_code" and file_path:
                header = f"[SOURCE CODE — {file_path}]"
            elif source_type == "readme":
                header = f"[README — {repo_name}]"
            elif source_type == "documentation":
                header = f"[DOCUMENTATION — {repo_name}]"
            elif source_type == "metadata":
                header = f"[REPOSITORY METADATA — {repo_name}]"
            elif source_type == "directory_tree":
                header = f"[DIRECTORY STRUCTURE — {repo_name}]"
            else:
                header = f"[{source_type.upper()} — {repo_name}]"

            formatted_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_parts)
