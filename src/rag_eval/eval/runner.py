"""Run the evaluation set and aggregate a scorecard.

For each labeled case: retrieve, measure retrieval, generate a cited answer,
and (optionally) score it with the LLM judge. The slow part — the answer and
judge LLM calls — runs concurrently across a bounded thread pool, so a 25-case
eval finishes in seconds instead of minutes. Retrieval runs sequentially
(it's fast and CPU-bound). A progress callback lets the CLI show live progress.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..answer import generate_answer
from ..config import RunConfig
from ..llm import get_llm_client
from ..observability import get_logger
from ..pipeline import build_retriever
from ..schemas import CaseResult, Scorecard
from .artifacts import save_scorecard
from .dataset import load_eval_set
from .judge import judge_answer
from .retrieval_metrics import compute_retrieval_metrics

_log = get_logger("eval")

ProgressFn = Callable[[int, int], None]  # (completed, total)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval(
    cfg: RunConfig,
    *,
    answers: bool = True,
    judge: bool = True,
    limit: int | None = None,
    save: bool | None = None,
    on_progress: ProgressFn | None = None,
) -> Scorecard:
    """Evaluate the pipeline defined by ``cfg`` over the labeled eval set.

    ``answers=False`` measures retrieval only (no LLM calls) — used by the
    config sweep. ``judge=False`` still generates answers but skips the judge.
    ``save`` overrides ``cfg.eval.save_runs`` for persisting the run.
    """
    cases = load_eval_set(cfg.paths.eval_set)
    if limit is not None:
        cases = cases[:limit]
    total = len(cases)

    retriever = build_retriever(cfg)
    answerer = get_llm_client(cfg.llm, cfg.retry) if answers else None
    judge_llm = get_llm_client(cfg.judge, cfg.retry) if (answers and judge) else None
    k = cfg.retrieval.top_k

    # Phase 1 — retrieval + deterministic metrics (sequential, fast).
    prepared = []
    for case in cases:
        retrieval = retriever.retrieve(case.question)
        metrics = compute_retrieval_metrics(retrieval, case.relevant_doc_ids, k)
        prepared.append((case, retrieval, metrics))

    results: list[CaseResult | None] = [None] * total

    def _process(idx: int) -> tuple[int, CaseResult]:
        case, retrieval, metrics = prepared[idx]
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
        return idx, CaseResult(
            case=case, retrieval_metrics=metrics, answer=answer, judgement=judgement
        )

    if answerer is None:
        # No LLM work; assemble results directly.
        for idx, (case, _r, metrics) in enumerate(prepared):
            results[idx] = CaseResult(case=case, retrieval_metrics=metrics)
            if on_progress:
                on_progress(idx + 1, total)
    else:
        # Phase 2 — answer + judge concurrently (network-bound).
        workers = max(1, min(cfg.eval.concurrency, total))
        _log.info("Scoring %d cases with %d concurrent workers", total, workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, i) for i in range(total)]
            done = 0
            for fut in as_completed(futures):
                idx, case_result = fut.result()
                results[idx] = case_result
                done += 1
                if on_progress:
                    on_progress(done, total)

    finalized = [r for r in results if r is not None]
    scorecard = _aggregate(cfg, finalized, k)

    should_save = cfg.eval.save_runs if save is None else save
    if should_save:
        path = save_scorecard(scorecard, cfg.paths.runs_dir)
        _log.info("Saved run to %s", path)

    return scorecard


def _aggregate(cfg: RunConfig, results: list[CaseResult], k: int) -> Scorecard:
    # Retrieval metrics are averaged only over answerable cases; unanswerable
    # cases are judged on answer quality (did the system correctly decline?).
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
