"""High-level pipeline entry points shared by the CLI and the eval harness.

Each function is a pure function of ``RunConfig`` — build the same components
the same way every time, so an eval run exercises exactly the pipeline a user
would get from ``ask``.
"""

from __future__ import annotations

from .answer import generate_answer
from .config import RunConfig
from .embed import get_embedding_model
from .ingest import chunk_document, load_corpus
from .llm import get_llm_client
from .retrieve import Retriever
from .schemas import Answer
from .store import VectorStore


def build_retriever(cfg: RunConfig) -> Retriever:
    """Construct a retriever from config (embedding provider + vector store)."""
    embedder = get_embedding_model(cfg.embedding)
    store = VectorStore(cfg.storage)
    return Retriever(store, embedder, cfg.retrieval.top_k)


def run_ingest(cfg: RunConfig, *, rebuild: bool = False) -> dict[str, int]:
    """Load the corpus, chunk it, embed it, and populate the vector store."""
    embedder = get_embedding_model(cfg.embedding)
    store = VectorStore(cfg.storage)
    if rebuild:
        store.reset()

    documents = load_corpus(cfg.paths.corpus_dir)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, cfg.chunking))

    if all_chunks:
        embeddings = embedder.embed_documents([c.text for c in all_chunks])
        store.add(all_chunks, embeddings)

    return {"documents": len(documents), "chunks": len(all_chunks)}


def run_ask(cfg: RunConfig, question: str) -> Answer:
    """Retrieve context and produce a cited answer for a single question."""
    retriever = build_retriever(cfg)
    llm = get_llm_client(cfg.llm)
    retrieval = retriever.retrieve(question)
    return generate_answer(llm, question, retrieval)
