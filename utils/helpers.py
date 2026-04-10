"""
utils/helpers.py
----------------------------------------------------------
Shared utilities — logging setup, text helpers, etc.
----------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys

from config import settings


def setup_logging() -> None:
    """
    Configure root logger with a clean, readable format.
    Log level is controlled via the LOG_LEVEL env variable.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate a string, appending '…' if it exceeds max_length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"
