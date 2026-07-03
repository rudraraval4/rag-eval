from rag_eval.eval.retrieval_metrics import compute_retrieval_metrics
from rag_eval.schemas import Chunk, RetrievalResult, RetrievedChunk


def _result(doc_ids: list[str]) -> RetrievalResult:
    retrieved = []
    for rank, doc_id in enumerate(doc_ids, start=1):
        chunk = Chunk(
            chunk_id=f"{doc_id}::{rank}",
            doc_id=doc_id,
            source_path=f"{doc_id}.txt",
            text="x",
            chunk_index=rank - 1,
            start_char=0,
            end_char=1,
        )
        retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 / rank, rank=rank))
    return RetrievalResult(query="q", retrieved=retrieved)


def test_perfect_retrieval_at_rank_one():
    m = compute_retrieval_metrics(_result(["a", "b", "c"]), ["a"], k=3)
    assert m.hit_rate == 1.0
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0
    assert m.precision_at_k == 1 / 3


def test_relevant_doc_lower_in_ranking():
    m = compute_retrieval_metrics(_result(["x", "y", "a"]), ["a"], k=3)
    assert m.hit_rate == 1.0
    assert m.mrr == 1 / 3


def test_miss_scores_zero():
    m = compute_retrieval_metrics(_result(["x", "y", "z"]), ["a"], k=3)
    assert m.hit_rate == 0.0
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0
    assert m.precision_at_k == 0.0


def test_recall_with_multiple_relevant_docs():
    m = compute_retrieval_metrics(_result(["a", "x", "b"]), ["a", "b", "c"], k=3)
    assert m.recall_at_k == 2 / 3  # retrieved a and b, missed c
    assert m.precision_at_k == 2 / 3


def test_k_truncates_retrieved_list():
    # Relevant doc is at rank 3 but k=2, so it should be excluded.
    m = compute_retrieval_metrics(_result(["x", "y", "a"]), ["a"], k=2)
    assert m.hit_rate == 0.0


def test_unanswerable_case_returns_zeros():
    m = compute_retrieval_metrics(_result(["x", "y"]), [], k=2)
    assert m.hit_rate == 0.0
    assert m.recall_at_k == 0.0
