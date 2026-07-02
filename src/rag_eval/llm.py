"""Pluggable LLM providers for answering and judging.

Default is Groq (fast, free tier). OpenAI and Anthropic (Claude) are opt-in:
their SDKs are lazy-imported and a missing key produces a clear error. The
interface is deliberately tiny — one single-turn ``complete(system, user)``
call covers both cited answering and LLM-as-judge scoring.

Add a provider by writing one class with ``complete`` and a ``name`` property
and registering it in :func:`get_llm_client`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import LLMConfig
from .errors import MissingDependencyError, require_env


@runtime_checkable
class LLMClient(Protocol):
    """Minimal single-turn chat interface."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


class GroqClient:
    """Groq chat completions (OpenAI-compatible). Requires GROQ_API_KEY."""

    def __init__(self, cfg: LLMConfig) -> None:
        api_key = require_env("groq", "GROQ_API_KEY")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - base dependency
            raise MissingDependencyError("groq", "groq", "all") from exc

        self._client = Groq(api_key=api_key)
        self._cfg = cfg
        self.name = f"groq:{cfg.model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class OpenAIClient:
    """OpenAI chat completions. Requires OPENAI_API_KEY."""

    def __init__(self, cfg: LLMConfig) -> None:
        api_key = require_env("openai", "OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai", "openai") from exc

        self._client = OpenAI(api_key=api_key)
        self._cfg = cfg
        self.name = f"openai:{cfg.model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    """Anthropic (Claude) messages API. Requires ANTHROPIC_API_KEY.

    Note: newer Claude models (Sonnet 5, Opus 4.8, ...) reject non-default
    sampling parameters, so ``temperature`` from config is intentionally NOT
    sent here — steer Claude via prompting instead. The ``system`` prompt is a
    top-level parameter, and the user turn carries the question + context.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        api_key = require_env("anthropic", "ANTHROPIC_API_KEY")
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic", "anthropic", "anthropic") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._cfg = cfg
        self.name = f"anthropic:{cfg.model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class MockLLM:
    """Deterministic stand-in for tests — echoes a canned or rule-based reply."""

    def __init__(self, reply: str = "", *, name: str = "mock:mock") -> None:
        self._reply = reply
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._reply


def get_llm_client(cfg: LLMConfig) -> LLMClient:
    """Factory: build the LLM client selected by config."""
    if cfg.provider == "groq":
        return GroqClient(cfg)
    if cfg.provider == "openai":
        return OpenAIClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")
