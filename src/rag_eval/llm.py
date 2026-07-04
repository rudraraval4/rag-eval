"""Pluggable LLM providers for answering and judging.

Default is Groq (fast, free tier). OpenAI and Anthropic (Claude) are opt-in:
their SDKs are lazy-imported and a missing key produces a clear error. The
interface is deliberately tiny — one single-turn ``complete(system, user)``
call covers both cited answering and LLM-as-judge scoring.

Every network call is wrapped with retry/backoff (transient rate-limits and
5xx are retried) and logs its latency and token usage for observability.

Add a provider by writing one class with ``complete`` and a ``name`` property
and registering it in :func:`get_llm_client`.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from .config import LLMConfig, RetryConfig
from .errors import MissingDependencyError, require_env
from .observability import get_logger
from .resilience import call_with_retry

_log = get_logger("llm")


@runtime_checkable
class LLMClient(Protocol):
    """Minimal single-turn chat interface."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


class _BaseLLM:
    """Shared retry + logging around a provider-specific ``_raw_complete``."""

    name: str

    def __init__(self, cfg: LLMConfig, retry: RetryConfig) -> None:
        self._cfg = cfg
        self._retry = retry

    def _raw_complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> str:
        start = time.perf_counter()
        text, usage = call_with_retry(
            lambda: self._raw_complete(system, user),
            self._retry,
            what=f"{self.name} completion",
        )
        elapsed = time.perf_counter() - start
        _log.debug(
            "%s completed in %.2fs (tokens: %s)",
            self.name,
            elapsed,
            usage or "n/a",
        )
        return text


class GroqClient(_BaseLLM):
    """Groq chat completions (OpenAI-compatible). Requires GROQ_API_KEY."""

    def __init__(self, cfg: LLMConfig, retry: RetryConfig) -> None:
        super().__init__(cfg, retry)
        api_key = require_env("groq", "GROQ_API_KEY")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - base dependency
            raise MissingDependencyError("groq", "groq", "all") from exc
        self._client = Groq(api_key=api_key)
        self.name = f"groq:{cfg.model}"

    def _raw_complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or "", _usage_dict(usage)


class OpenAIClient(_BaseLLM):
    """OpenAI chat completions. Requires OPENAI_API_KEY."""

    def __init__(self, cfg: LLMConfig, retry: RetryConfig) -> None:
        super().__init__(cfg, retry)
        api_key = require_env("openai", "OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai", "openai") from exc
        self._client = OpenAI(api_key=api_key)
        self.name = f"openai:{cfg.model}"

    def _raw_complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or "", _usage_dict(
            getattr(resp, "usage", None)
        )


class AnthropicClient(_BaseLLM):
    """Anthropic (Claude) messages API. Requires ANTHROPIC_API_KEY.

    Note: newer Claude models (Sonnet 5, Opus 4.8, ...) reject non-default
    sampling parameters, so ``temperature`` from config is intentionally NOT
    sent here — steer Claude via prompting instead. The ``system`` prompt is a
    top-level parameter, and the user turn carries the question + context.
    """

    def __init__(self, cfg: LLMConfig, retry: RetryConfig) -> None:
        super().__init__(cfg, retry)
        api_key = require_env("anthropic", "ANTHROPIC_API_KEY")
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic", "anthropic", "anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.name = f"anthropic:{cfg.model}"

    def _raw_complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        resp = self._client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        usage = getattr(resp, "usage", None)
        tokens = (
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
            if usage
            else {}
        )
        return text, tokens


class MockLLM:
    """Deterministic stand-in for tests — echoes a canned or rule-based reply."""

    def __init__(self, reply: str = "", *, name: str = "mock:mock") -> None:
        self._reply = reply
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._reply


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def get_llm_client(cfg: LLMConfig, retry: RetryConfig | None = None) -> LLMClient:
    """Factory: build the LLM client selected by config."""
    retry = retry or RetryConfig()
    if cfg.provider == "groq":
        return GroqClient(cfg, retry)
    if cfg.provider == "openai":
        return OpenAIClient(cfg, retry)
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg, retry)
    raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")
