"""Rich terminal rendering for answers, scorecards, and sweep tables."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .schemas import Answer, Scorecard


def print_answer(console: Console, answer: Answer) -> None:
    console.print(Panel(answer.text, title="Answer", border_style="green"))

    if answer.citations:
        table = Table(title="Citations", show_lines=False)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Source", style="magenta")
        table.add_column("Chunk", style="dim")
        for c in answer.citations:
            table.add_row(str(c.marker), c.doc_id, c.chunk_id)
        console.print(table)
    else:
        console.print("[yellow]No citations resolved for this answer.[/]")

    console.print(f"[dim]model: {answer.model} · retrieved: {len(answer.retrieval.retrieved)} chunks[/]")


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def print_scorecard(console: Console, sc: Scorecard) -> None:
    cfg = sc.config_summary
    header = " · ".join(f"{k}={v}" for k, v in cfg.items())
    console.print(Panel(header, title="Run configuration", border_style="blue"))

    ret = Table(title=f"Retrieval metrics (k={sc.k}, n={sc.num_cases})")
    ret.add_column("Metric", style="cyan")
    ret.add_column("Score", justify="right")
    ret.add_row("hit_rate", _fmt(sc.hit_rate))
    ret.add_row(f"recall@{sc.k}", _fmt(sc.recall_at_k))
    ret.add_row(f"precision@{sc.k}", _fmt(sc.precision_at_k))
    ret.add_row("MRR", _fmt(sc.mrr))
    console.print(ret)

    ans = Table(title="Answer-quality metrics (LLM-as-judge)")
    ans.add_column("Metric", style="cyan")
    ans.add_column("Score", justify="right")
    ans.add_row("faithfulness", _fmt(sc.faithfulness))
    ans.add_row("answer_relevance", _fmt(sc.answer_relevance))
    ans.add_row("hallucination_rate", _fmt(sc.hallucination_rate))
    console.print(ans)


def print_sweep(console: Console, results: list[Scorecard]) -> None:
    table = Table(title="Configuration comparison", show_lines=True)
    table.add_column("chunk", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("k", justify="right")
    table.add_column("embedding")
    table.add_column("recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("faithful", justify="right")
    table.add_column("halluc.", justify="right")

    for sc in results:
        cfg = sc.config_summary
        table.add_row(
            cfg.get("chunk_size", "—"),
            cfg.get("chunk_overlap", "—"),
            cfg.get("top_k", "—"),
            cfg.get("embedding", "—"),
            _fmt(sc.recall_at_k),
            _fmt(sc.mrr),
            _fmt(sc.faithfulness),
            _fmt(sc.hallucination_rate),
        )
    console.print(table)
