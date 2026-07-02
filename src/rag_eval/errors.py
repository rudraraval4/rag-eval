"""Friendly, actionable errors for common misconfiguration."""

from __future__ import annotations

import os


class RagEvalError(Exception):
    """Base class for all rag-eval errors."""


class MissingAPIKeyError(RagEvalError):
    """Raised when a selected provider needs an API key that isn't set."""

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(
            f"Provider '{provider}' is selected but {env_var} is not set. "
            f"Add it to your environment or .env file (see .env.example)."
        )


class MissingDependencyError(RagEvalError):
    """Raised when an optional provider's SDK is not installed."""

    def __init__(self, provider: str, package: str, extra: str) -> None:
        super().__init__(
            f"Provider '{provider}' requires the '{package}' package, which is not installed. "
            f"Install it with: pip install 'rag-eval[{extra}]'"
        )


def require_env(provider: str, env_var: str) -> str:
    """Return the value of ``env_var`` or raise a friendly error."""
    value = os.getenv(env_var)
    if not value:
        raise MissingAPIKeyError(provider, env_var)
    return value
