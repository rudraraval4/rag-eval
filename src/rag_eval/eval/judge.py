"""LLM-as-judge for answer quality.

A separate (and ideally stronger / different) model scores each answer against
the retrieved context. Prompts are pinned and the judge runs at temperature 0
for reproducibility. We score three things:

- faithfulness:     is every claim supported by the retrieved context? [0,1]
- answer_relevance: does the answer actually address the question? [0,1]
- hallucination:    does the answer assert anything the context doesn't support?

The judge is asked for strict JSON; we parse defensively so a stray sentence
around the JSON doesn't crash a whole eval run.
"""

from __future__ import annotations

import json
import re

from ..llm import LLMClient
from ..schemas import AnswerJudgement, RetrievalResult

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of retrieval-augmented answers. "
    "You will be given a question, the context passages that were retrieved, and "
    "an answer. Judge the answer ONLY against the provided context, not your own "
    "knowledge. Respond with a single JSON object and nothing else, with keys:\n"
    '  "faithfulness": number 0-1 (1 = every claim is supported by the context),\n'
    '  "answer_relevance": number 0-1 (1 = fully addresses the question),\n'
    '  "hallucination": boolean (true if the answer asserts anything not '
    "supported by the context),\n"
    '  "rationale": a one-sentence explanation.\n'
    "A correct refusal to answer when the context lacks the information is "
    "faithful and non-hallucinated."
)


def _parse_judgement(raw: str) -> AnswerJudgement:
    match = _JSON_RE.search(raw)
    if not match:
        return AnswerJudgement(
            faithfulness=0.0,
            answer_relevance=0.0,
            hallucination=True,
            rationale="Could not parse judge output.",
        )
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return AnswerJudgement(
            faithfulness=0.0,
            answer_relevance=0.0,
            hallucination=True,
            rationale="Invalid JSON from judge.",
        )

    def _clamp(x: object) -> float:
        try:
            return max(0.0, min(1.0, float(x)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    return AnswerJudgement(
        faithfulness=_clamp(data.get("faithfulness")),
        answer_relevance=_clamp(data.get("answer_relevance")),
        hallucination=bool(data.get("hallucination", False)),
        rationale=str(data.get("rationale", "")),
    )


def judge_answer(
    judge: LLMClient, question: str, answer_text: str, retrieval: RetrievalResult
) -> AnswerJudgement:
    """Score a single answer with the LLM judge."""
    context = "\n\n".join(
        f"[{rc.rank}] {rc.chunk.text}" for rc in retrieval.retrieved
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer to evaluate:\n{answer_text}\n\n"
        "Return the JSON object now."
    )
    raw = judge.complete(JUDGE_SYSTEM_PROMPT, user_prompt)
    return _parse_judgement(raw)
