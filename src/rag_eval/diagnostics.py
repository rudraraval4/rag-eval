"""Preflight checks for the ``doctor`` command.

Answers "will this actually run?" before a user burns time on a failed ingest
or eval: are the selected providers' API keys present, does the corpus exist, is
the index built, is the eval set valid. Returns structured results the CLI
renders as a checklist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig

# Which env var each provider needs (None = no key required).
_EMBED_KEYS = {"bge": None, "openai": "OPENAI_API_KEY", "voyage": "VOYAGE_API_KEY"}
_LLM_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _key_check(label: str, provider: str, env_var: str | None) -> Check:
    if env_var is None:
        return Check(label, True, f"{provider}: no API key required")
    if os.getenv(env_var):
        return Check(label, True, f"{provider}: {env_var} is set")
    return Check(label, False, f"{provider}: {env_var} is NOT set")


def run_diagnostics(cfg: RunConfig) -> list[Check]:
    """Run all preflight checks for the given config."""
    checks: list[Check] = []

    # Provider keys for the selected providers.
    checks.append(
        _key_check("embedding key", cfg.embedding.provider, _EMBED_KEYS.get(cfg.embedding.provider))
    )
    checks.append(_key_check("answer llm key", cfg.llm.provider, _LLM_KEYS.get(cfg.llm.provider)))
    checks.append(_key_check("judge llm key", cfg.judge.provider, _LLM_KEYS.get(cfg.judge.provider)))

    # Corpus.
    corpus = Path(cfg.paths.corpus_dir)
    n_docs = (
        sum(1 for p in corpus.rglob("*") if p.is_file()) if corpus.exists() else 0
    )
    checks.append(
        Check("corpus", corpus.exists() and n_docs > 0, f"{corpus} ({n_docs} files)")
    )

    # Eval set.
    eval_path = Path(cfg.paths.eval_set)
    checks.append(Check("eval set", eval_path.exists(), str(eval_path)))

    # Vector index.
    try:
        from .store import VectorStore

        count = VectorStore(cfg.storage).count()
        checks.append(
            Check(
                "vector index",
                count > 0,
                f"{cfg.storage.collection}: {count} chunks (run `ingest` if 0)",
            )
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the doctor
        checks.append(Check("vector index", False, f"error: {exc}"))

    return checks
