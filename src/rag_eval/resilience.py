"""Retry with exponential backoff for transient provider failures.

Hosted providers (Groq, OpenAI, Anthropic, Voyage) rate-limit and occasionally
5xx. Rather than let a single 429 abort a whole eval run, we retry transient
errors with jittered exponential backoff. Non-transient errors (bad key, bad
request) are raised immediately — retrying them would just waste time.

Provider SDKs don't share a base exception, so transience is detected
structurally: an HTTP status of 429 or ≥500, or an error class whose name marks
it as a rate-limit / connection / timeout error.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from .config import RetryConfig
from .observability import get_logger

T = TypeVar("T")
_log = get_logger("resilience")

_TRANSIENT_NAME_HINTS = (
    "RateLimit",
    "APIConnection",
    "Connection",
    "Timeout",
    "InternalServer",
    "ServiceUnavailable",
    "Overloaded",
)


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient(exc: BaseException) -> bool:
    """Heuristically decide whether an error is worth retrying."""
    status = _status_of(exc)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__
    return any(hint in name for hint in _TRANSIENT_NAME_HINTS)


def call_with_retry(
    fn: Callable[[], T], cfg: RetryConfig, *, what: str = "provider call"
) -> T:
    """Call ``fn`` with retries on transient errors. Returns its result."""
    last: BaseException | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified by is_transient
            last = exc
            if not is_transient(exc) or attempt == cfg.max_attempts:
                raise
            delay = min(cfg.base_delay * (2 ** (attempt - 1)), cfg.max_delay)
            delay += random.uniform(0, cfg.base_delay)  # jitter avoids thundering herd
            _log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what,
                attempt,
                cfg.max_attempts,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    # Unreachable, but keeps type-checkers happy.
    raise last  # type: ignore[misc]
