"""Retrieval: embed a query and fetch the top-k chunks.

A tiny orchestration layer so answering and evaluation share the exact same
retrieval path — the eval harness measures precisely what ``ask`` uses.
"""

from __future__ import annotations

from .embed import EmbeddingModel
from .schemas import RetrievalResult
from .store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingModel, top_k: int) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self._top_k
        query_vec = self._embedder.embed_query(query)
        retrieved = self._store.query(query_vec, k)
        return RetrievalResult(query=query, retrieved=retrieved)
