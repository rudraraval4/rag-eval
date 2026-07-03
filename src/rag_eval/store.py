"""Chroma-backed vector store.

Thin wrapper over a persistent Chroma collection. We compute embeddings
ourselves (via the pluggable embedding provider) and hand Chroma the vectors
directly, so the store is agnostic to which embedding model produced them.
Cosine space is used because all providers here return normalized vectors.

Chunk text is stored as the Chroma "document" and structural fields as
metadata, so retrieved chunks reconstruct into the exact same ``Chunk`` model —
that round-trip is what lets citations point at real source spans.
"""

from __future__ import annotations

from .config import StorageConfig
from .schemas import Chunk, RetrievedChunk

_RESERVED_KEYS = {"doc_id", "source_path", "chunk_index", "start_char", "end_char"}


class VectorStore:
    def __init__(self, cfg: StorageConfig) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - base dependency
            from .errors import MissingDependencyError

            raise MissingDependencyError("chroma", "chromadb", "all") from exc

        self._client = chromadb.PersistentClient(path=cfg.persist_dir)
        self._cfg = cfg
        self._collection = self._client.get_or_create_collection(
            name=cfg.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection (used by ``ingest --rebuild``)."""
        try:
            self._client.delete_collection(self._cfg.collection)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._cfg.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their precomputed embeddings to the collection."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        metadatas = []
        for c in chunks:
            meta: dict[str, str | int | float | bool] = {
                "doc_id": c.doc_id,
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            # Preserve extra document metadata that Chroma can store.
            for k, v in c.metadata.items():
                if k not in _RESERVED_KEYS and isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks for a query embedding."""
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        retrieved: list[RetrievedChunk] = []
        for rank, (cid, text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            extra = {
                k: str(v) for k, v in meta.items() if k not in _RESERVED_KEYS
            }
            chunk = Chunk(
                chunk_id=cid,
                doc_id=str(meta["doc_id"]),
                source_path=str(meta["source_path"]),
                text=text,
                chunk_index=int(meta["chunk_index"]),
                start_char=int(meta["start_char"]),
                end_char=int(meta["end_char"]),
                metadata=extra,
            )
            # Cosine distance in [0, 2] → similarity score (higher is better).
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - dist, rank=rank))
        return retrieved
