"""Pluggable embedding providers.

Default is local BGE (no API key, fully offline, reproducible). OpenAI and
Voyage are opt-in: their SDKs are lazy-imported so the base install stays lean,
and a missing key produces a clear error rather than an obscure stack trace.

Add a provider by writing one class with ``embed_documents`` / ``embed_query``
and registering it in :func:`get_embedding_model`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import EmbeddingConfig
from .errors import MissingDependencyError, require_env

# BGE v1.5 models were trained with a query-side instruction for retrieval.
# Documents are embedded as-is; only the query gets the prefix.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class EmbeddingModel(Protocol):
    """Minimal embedding interface. One method to embed docs, one for queries."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BGEEmbedding:
    """Local sentence-transformers BGE model. No API key required."""

    def __init__(self, model: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - base dependency
            raise MissingDependencyError("bge", "sentence-transformers", "all") from exc

        self._model = SentenceTransformer(model)
        self._batch_size = batch_size
        self._is_bge = "bge" in model.lower()
        # sentence-transformers renamed this method; support both versions.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        query = f"{_BGE_QUERY_INSTRUCTION}{text}" if self._is_bge else text
        vector = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vector.tolist()


class OpenAIEmbedding:
    """Hosted OpenAI embeddings. Requires OPENAI_API_KEY."""

    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str, batch_size: int = 128) -> None:
        api_key = require_env("openai", "OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai", "openai") from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self.dim = self._DIMS.get(model, 1536)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class VoyageEmbedding:
    """Hosted Voyage embeddings. Requires VOYAGE_API_KEY."""

    _DIMS = {"voyage-3": 1024, "voyage-3-lite": 512}

    def __init__(self, model: str, batch_size: int = 128) -> None:
        api_key = require_env("voyage", "VOYAGE_API_KEY")
        try:
            import voyageai
        except ImportError as exc:
            raise MissingDependencyError("voyage", "voyageai", "voyage") from exc

        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self.dim = self._DIMS.get(model, 1024)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = self._client.embed(batch, model=self._model, input_type="document")
            out.extend(resp.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embed([text], model=self._model, input_type="query")
        return resp.embeddings[0]


def get_embedding_model(cfg: EmbeddingConfig) -> EmbeddingModel:
    """Factory: build the embedding model selected by config."""
    if cfg.provider == "bge":
        return BGEEmbedding(cfg.model, cfg.batch_size)
    if cfg.provider == "openai":
        return OpenAIEmbedding(cfg.model, cfg.batch_size)
    if cfg.provider == "voyage":
        return VoyageEmbedding(cfg.model, cfg.batch_size)
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
