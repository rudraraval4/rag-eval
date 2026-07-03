"""Run the evaluation set and aggregate a scorecard.

For each labeled case: retrieve, measure retrieval, generate a cited answer,
and (optionally) score it with the LLM judge. Aggregates are means over the
cases, so the scorecard is a single, comparable summary of one configuration.
"""

from __future__ import annotations

from ..answer import generate_answer
from ..config import RunConfig
from ..llm import get_llm_client
from ..pipeline import build_retriever
from ..schemas import CaseResult, Scorecard
from .dataset import load_eval_set
from .judge import judge_answer
from .retrieval_metrics import compute_retrieval_metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval(
    cfg: RunConfig,
    *,
    answers: bool = True,
    judge: bool = True,
    limit: int | None = None,
) -> Scorecard:
    """Evaluate the pipeline defined by ``cfg`` over the labeled eval set.

    ``answers=False`` measures retrieval only (no LLM calls) — used by the
    config sweep, where the deterministic retrieval metrics are what chunk size
    and k actually move. ``judge=False`` still generates answers but skips the
    LLM judge.
    """
    cases = load_eval_set(cfg.paths.eval_set)
    if limit is not None:
        cases = cases[:limit]

    retriever = build_retriever(cfg)
    answerer = get_llm_client(cfg.llm) if answers else None
    judge_llm = get_llm_client(cfg.judge) if (answers and judge) else None
    k = cfg.retrieval.top_k

    results: list[CaseResult] = []
    for case in cases:
        retrieval = retriever.retrieve(case.question)
        metrics = compute_retrieval_metrics(retrieval, case.relevant_doc_ids, k)
        answer = (
            generate_answer(answerer, case.question, retrieval)
            if answerer is not None
            else None
        )
        judgement = (
            judge_answer(judge_llm, case.question, answer.text, retrieval)
            if judge_llm is not None and answer is not None
            else None
        )
        results.append(
            CaseResult(
                case=case,
                retrieval_metrics=metrics,
                answer=answer,
                judgement=judgement,
            )
        )

    # Retrieval metrics are averaged only over answerable cases (those with
    # ground-truth documents); unanswerable cases are judged on answer quality.
    answerable = [r for r in results if r.case.relevant_doc_ids]
    judged = [r for r in results if r.judgement is not None]

    return Scorecard(
        config_summary=cfg.summary(),
        num_cases=len(results),
        k=k,
        hit_rate=_mean([r.retrieval_metrics.hit_rate for r in answerable]),
        recall_at_k=_mean([r.retrieval_metrics.recall_at_k for r in answerable]),
        precision_at_k=_mean([r.retrieval_metrics.precision_at_k for r in answerable]),
        mrr=_mean([r.retrieval_metrics.mrr for r in answerable]),
        faithfulness=(
            _mean([r.judgement.faithfulness for r in judged]) if judged else None
        ),
        answer_relevance=(
            _mean([r.judgement.answer_relevance for r in judged]) if judged else None
        ),
        hallucination_rate=(
            _mean([1.0 if r.judgement.hallucination else 0.0 for r in judged])
            if judged
            else None
        ),
        results=results,
    )
