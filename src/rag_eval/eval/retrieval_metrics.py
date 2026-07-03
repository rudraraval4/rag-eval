"""Deterministic retrieval metrics.

No LLM involved — these are exact, reproducible measurements of whether the
retriever fetched the right source documents. Ground truth is at the document
level (an eval case lists the ``relevant_doc_ids`` that should be retrieved),
so metrics are computed over the distinct documents present in the retrieved
chunks.

- hit_rate:     did we retrieve at least one relevant document? (0 or 1)
- recall@k:     fraction of relevant documents that were retrieved
- precision@k:  fraction of retrieved chunks belonging to a relevant document
- MRR:          reciprocal rank of the first chunk from a relevant document
"""

from __future__ import annotations

from ..schemas import RetrievalMetrics, RetrievalResult


def compute_retrieval_metrics(
    retrieval: RetrievalResult, relevant_doc_ids: list[str], k: int
) -> RetrievalMetrics:
    """Compute retrieval metrics for a single query against its ground truth."""
    relevant = set(relevant_doc_ids)
    retrieved_chunks = retrieval.retrieved[:k]
    n_retrieved = len(retrieved_chunks)

    if not relevant:
        # No ground-truth documents (e.g. an unanswerable question) — retrieval
        # metrics are undefined; report zeros and let answer-quality carry it.
        return RetrievalMetrics(
            hit_rate=0.0, recall_at_k=0.0, precision_at_k=0.0, mrr=0.0, k=k
        )

    retrieved_relevant_docs = {
        rc.chunk.doc_id for rc in retrieved_chunks if rc.chunk.doc_id in relevant
    }
    hit_rate = 1.0 if retrieved_relevant_docs else 0.0
    recall = len(retrieved_relevant_docs) / len(relevant)
    relevant_chunk_count = sum(
        1 for rc in retrieved_chunks if rc.chunk.doc_id in relevant
    )
    precision = relevant_chunk_count / n_retrieved if n_retrieved else 0.0

    mrr = 0.0
    for rc in retrieved_chunks:
        if rc.chunk.doc_id in relevant:
            mrr = 1.0 / rc.rank
            break

    return RetrievalMetrics(
        hit_rate=hit_rate,
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        k=k,
    )
