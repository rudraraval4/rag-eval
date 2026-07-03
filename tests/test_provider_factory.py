import pytest

from rag_eval.config import EmbeddingConfig, LLMConfig
from rag_eval.embed import get_embedding_model
from rag_eval.errors import MissingAPIKeyError
from rag_eval.llm import get_llm_client


def test_llm_missing_key_raises_friendly_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_llm_client(LLMConfig(provider="groq", model="llama-3.3-70b-versatile"))


def test_openai_llm_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_llm_client(LLMConfig(provider="openai", model="gpt-4o-mini"))


def test_anthropic_llm_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_llm_client(LLMConfig(provider="anthropic", model="claude-sonnet-5"))


def test_embedding_missing_key_raises(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_embedding_model(EmbeddingConfig(provider="voyage", model="voyage-3"))


def test_unknown_llm_provider_rejected():
    cfg = LLMConfig.model_construct(provider="nope", model="x", temperature=0.0, max_tokens=10)
    with pytest.raises(ValueError):
        get_llm_client(cfg)
