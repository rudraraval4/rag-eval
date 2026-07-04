"""Persist evaluation runs to disk.

Every scored run is written to ``runs/<timestamp>/`` as machine-readable JSON
(full per-case detail) and a human-readable Markdown scorecard. This makes
results auditable, diffable across commits, and easy to paste into a report —
the difference between "trust me, it's good" and "here's the run."
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..schemas import Scorecard


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def scorecard_to_markdown(sc: Scorecard) -> str:
    cfg = sc.config_summary
    lines = [
        "# Evaluation scorecard",
        "",
        f"- Generated: {sc.created_at.isoformat()}",
        f"- Cases: {sc.num_cases}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key, val in cfg.items():
        lines.append(f"| {key} | {val} |")
    lines += [
        "",
        f"## Retrieval metrics (k={sc.k})",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| hit_rate | {_fmt(sc.hit_rate)} |",
        f"| recall@{sc.k} | {_fmt(sc.recall_at_k)} |",
        f"| precision@{sc.k} | {_fmt(sc.precision_at_k)} |",
        f"| MRR | {_fmt(sc.mrr)} |",
        "",
        "## Answer-quality metrics (LLM-as-judge)",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| faithfulness | {_fmt(sc.faithfulness)} |",
        f"| answer_relevance | {_fmt(sc.answer_relevance)} |",
        f"| hallucination_rate | {_fmt(sc.hallucination_rate)} |",
        "",
    ]
    return "\n".join(lines)


def save_scorecard(sc: Scorecard, runs_dir: str | Path) -> Path:
    """Write a run's JSON + Markdown into ``runs_dir/<timestamp>/`` and return it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(runs_dir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(sc.model_dump_json(indent=2), encoding="utf-8")
    (out / "scorecard.md").write_text(scorecard_to_markdown(sc), encoding="utf-8")
    return out
