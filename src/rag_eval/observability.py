"""Logging configuration.

One place to set up logging so every module — CLI, library, or API — emits
consistent, timestamped logs. Console verbosity is controllable; a rotating
file handler under ``logs/`` keeps a durable record of provider calls,
latencies, and retries for debugging production issues after the fact.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "rag_eval"
_configured = False


def configure_logging(
    *,
    level: int = logging.INFO,
    log_dir: str | Path = "logs",
    to_file: bool = True,
) -> logging.Logger:
    """Configure the package logger once and return it. Idempotent."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # handlers filter; logger passes everything

    if _configured:
        # Allow the console level to be adjusted on repeat calls.
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, RotatingFileHandler
            ):
                h.setLevel(level)
        return logger

    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(console)

    if to_file:
        try:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path / "rag_eval.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
        except OSError:
            # A read-only filesystem shouldn't break the whole run.
            logger.warning("Could not open log file; continuing with console only.")

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the package logger."""
    base = logging.getLogger(LOGGER_NAME)
    return base.getChild(name) if name else base
