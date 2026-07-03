from rag_eval.eval.judge import judge_answer
from rag_eval.llm import MockLLM
from rag_eval.schemas import Chunk, RetrievalResult, RetrievedChunk


def _retrieval() -> RetrievalResult:
    chunk = Chunk(
        chunk_id="d::0",
        doc_id="d",
        source_path="d.md",
        text="Paris is the capital of France.",
        chunk_index=0,
        start_char=0,
        end_char=31,
    )
    return RetrievalResult(query="q", retrieved=[RetrievedChunk(chunk=chunk, score=1.0, rank=1)])


def test_parses_clean_json():
    judge = MockLLM(
        '{"faithfulness": 1.0, "answer_relevance": 0.9, '
        '"hallucination": false, "rationale": "supported"}'
    )
    j = judge_answer(judge, "capital?", "Paris [1].", _retrieval())
    assert j.faithfulness == 1.0
    assert j.answer_relevance == 0.9
    assert j.hallucination is False
    assert j.rationale == "supported"


def test_extracts_json_embedded_in_prose():
    judge = MockLLM(
        'Here is my evaluation:\n{"faithfulness": 0.5, "answer_relevance": 0.5, '
        '"hallucination": true, "rationale": "partly"}\nDone.'
    )
    j = judge_answer(judge, "q", "a", _retrieval())
    assert j.hallucination is True
    assert j.faithfulness == 0.5


def test_clamps_out_of_range_scores():
    judge = MockLLM(
        '{"faithfulness": 5, "answer_relevance": -2, "hallucination": false}'
    )
    j = judge_answer(judge, "q", "a", _retrieval())
    assert j.faithfulness == 1.0
    assert j.answer_relevance == 0.0


def test_unparseable_output_fails_safe():
    judge = MockLLM("I cannot produce JSON.")
    j = judge_answer(judge, "q", "a", _retrieval())
    assert j.hallucination is True
    assert j.faithfulness == 0.0
