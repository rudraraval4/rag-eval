"""Rich terminal rendering for answers, scorecards, and sweep tables."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .schemas import Answer, Scorecard


@contextmanager
def progress_callback(
    console: Console, description: str
) -> Iterator[Callable[[int, int], None]]:
    """Yield an ``on_progress(done, total)`` callback backed by a live bar.

    The bar shows a spinner, an N/total counter, elapsed time, and an ETA — so a
    non-technical user can see it's working and roughly how long is left.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        state: dict[str, object] = {"task": None}

        def callback(done: int, total: int) -> None:
            if state["task"] is None:
                state["task"] = progress.add_task(description, total=total)
            progress.update(state["task"], completed=done)  # type: ignore[arg-type]

        yield callback


def print_doctor(console: Console, checks: list) -> bool:
    """Render preflight checks as a table. Returns True if all passed."""
    table = Table(title="rag-eval doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail", style="dim")
    all_ok = True
    for c in checks:
        mark = "[green]OK[/]" if c.ok else "[red]FAIL[/]"
        all_ok = all_ok and c.ok
        table.add_row(c.name, mark, c.detail)
    console.print(table)
    if all_ok:
        console.print("[green]All checks passed — you're ready to run.[/]")
    else:
        console.print("[yellow]Some checks failed — see details above.[/]")
    return all_ok


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

    n = len(answer.retrieval.retrieved)
    console.print(f"[dim]model: {answer.model} · retrieved: {n} chunks[/]")


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
    judged = any(sc.faithfulness is not None for sc in results)
    table = Table(title="Configuration comparison", show_lines=True)
    table.add_column("chunk", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("k", justify="right")
    table.add_column("hit_rate", justify="right")
    table.add_column("recall@k", justify="right")
    table.add_column("precision@k", justify="right")
    table.add_column("MRR", justify="right")
    if judged:
        table.add_column("faithful", justify="right")
        table.add_column("halluc.", justify="right")

    # Highlight the row with the best MRR (ties broken by recall).
    best = max(results, key=lambda s: (s.mrr, s.recall_at_k), default=None)

    for sc in results:
        cfg = sc.config_summary
        row = [
            cfg.get("chunk_size", "—"),
            cfg.get("chunk_overlap", "—"),
            cfg.get("top_k", "—"),
            _fmt(sc.hit_rate),
            _fmt(sc.recall_at_k),
            _fmt(sc.precision_at_k),
            _fmt(sc.mrr),
        ]
        if judged:
            row += [_fmt(sc.faithfulness), _fmt(sc.hallucination_rate)]
        style = "bold green" if sc is best else None
        table.add_row(*row, style=style)
    console.print(table)
