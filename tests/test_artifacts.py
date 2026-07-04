import json

from rag_eval.eval.artifacts import save_scorecard, scorecard_to_markdown
from rag_eval.schemas import Scorecard


def _scorecard() -> Scorecard:
    return Scorecard(
        config_summary={"embedding": "bge:x", "llm": "groq:y", "top_k": "5"},
        num_cases=25,
        k=5,
        hit_rate=1.0,
        recall_at_k=1.0,
        precision_at_k=0.217,
        mrr=0.893,
        faithfulness=0.96,
        answer_relevance=0.86,
        hallucination_rate=0.0,
    )


def test_markdown_contains_key_sections():
    md = scorecard_to_markdown(_scorecard())
    assert "# Evaluation scorecard" in md
    assert "recall@5" in md
    assert "faithfulness" in md
    assert "bge:x" in md


def test_save_writes_json_and_markdown(tmp_path):
    out = save_scorecard(_scorecard(), tmp_path)
    assert (out / "scorecard.json").exists()
    assert (out / "scorecard.md").exists()

    data = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    assert data["num_cases"] == 25
    assert data["mrr"] == 0.893
