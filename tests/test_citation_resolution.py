from rag_eval.answer import generate_answer
from rag_eval.llm import MockLLM
from rag_eval.schemas import Chunk, RetrievalResult, RetrievedChunk


def _retrieval(n: int) -> RetrievalResult:
    retrieved = []
    for rank in range(1, n + 1):
        chunk = Chunk(
            chunk_id=f"doc{rank}::0",
            doc_id=f"doc{rank}",
            source_path=f"doc{rank}.md",
            text=f"content {rank}",
            chunk_index=0,
            start_char=0,
            end_char=9,
        )
        retrieved.append(RetrievedChunk(chunk=chunk, score=1.0, rank=rank))
    return RetrievalResult(query="q", retrieved=retrieved)


def test_valid_citations_resolve_to_chunks():
    llm = MockLLM("The sky is blue [1] and grass is green [2].")
    ans = generate_answer(llm, "why?", _retrieval(3))
    assert [c.marker for c in ans.citations] == [1, 2]
    assert ans.citations[0].chunk_id == "doc1::0"
    assert ans.citations[1].doc_id == "doc2"


def test_fabricated_citations_are_dropped():
    # Marker [9] has no corresponding retrieved chunk and must be ignored.
    llm = MockLLM("Grounded [1] but fabricated [9].")
    ans = generate_answer(llm, "q", _retrieval(3))
    assert [c.marker for c in ans.citations] == [1]


def test_duplicate_markers_deduplicated_in_order():
    llm = MockLLM("Claim [2] and again [2] and also [1].")
    ans = generate_answer(llm, "q", _retrieval(3))
    assert [c.marker for c in ans.citations] == [2, 1]


def test_answer_records_model_name():
    llm = MockLLM("No citations here.", name="mock:test-model")
    ans = generate_answer(llm, "q", _retrieval(2))
    assert ans.model == "mock:test-model"
    assert ans.citations == []
