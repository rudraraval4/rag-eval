"""Command-line interface: ingest, ask, eval, sweep.

Thin wiring only — each command loads a RunConfig, applies CLI overrides, and
delegates to the pipeline modules. Business logic lives in the library so it
stays testable without the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from .config import RunConfig

app = typer.Typer(
    add_completion=False,
    help="A production-grade RAG pipeline with a first-class evaluation harness.",
)
console = Console()

DEFAULT_CONFIG = "configs/default.yaml"


def _load_config(config_path: str, **overrides: object) -> RunConfig:
    """Load config from YAML (falling back to defaults) and apply CLI overrides."""
    load_dotenv()  # pull provider API keys from .env if present
    if Path(config_path).exists():
        cfg = RunConfig.from_yaml(config_path)
    else:
        console.print(f"[yellow]Config {config_path} not found; using built-in defaults.[/]")
        cfg = RunConfig()
    return cfg.with_overrides(**overrides)


@app.command()
def ingest(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
    corpus_dir: Optional[str] = typer.Option(None, help="Override the corpus directory."),
    rebuild: bool = typer.Option(False, help="Drop and rebuild the collection from scratch."),
) -> None:
    """Build the vector index from a folder of documents."""
    cfg = _load_config(config, **{"paths.corpus_dir": corpus_dir})
    from .pipeline import run_ingest

    stats = run_ingest(cfg, rebuild=rebuild)
    console.print(
        f"[green]Ingested[/] {stats['documents']} documents "
        f"→ {stats['chunks']} chunks into '{cfg.storage.collection}'."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
    top_k: Optional[int] = typer.Option(None, help="Override number of chunks to retrieve."),
    llm_provider: Optional[str] = typer.Option(None, help="Override LLM provider."),
    llm_model: Optional[str] = typer.Option(None, help="Override LLM model."),
) -> None:
    """Answer a question with inline citations to source chunks."""
    cfg = _load_config(
        config,
        **{
            "retrieval.top_k": top_k,
            "llm.provider": llm_provider,
            "llm.model": llm_model,
        },
    )
    from .pipeline import run_ask
    from .render import print_answer

    answer = run_ask(cfg, question)
    print_answer(console, answer)


@app.command()
def eval(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
    no_judge: bool = typer.Option(False, help="Skip LLM-as-judge; retrieval metrics only."),
    limit: Optional[int] = typer.Option(None, help="Only evaluate the first N cases."),
) -> None:
    """Run the evaluation suite and print a scorecard."""
    cfg = _load_config(config)
    from .eval.runner import run_eval
    from .render import print_scorecard

    scorecard = run_eval(cfg, judge=not no_judge, limit=limit)
    print_scorecard(console, scorecard)


@app.command()
def sweep(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the base run config YAML."),
    with_answers: bool = typer.Option(
        False, help="Also score answer quality per config (many more LLM calls)."
    ),
) -> None:
    """Compare configurations (chunk size × k) and print a results table."""
    cfg = _load_config(config)
    from .eval.sweep import run_sweep
    from .render import print_sweep

    results = run_sweep(cfg, answers=with_answers, judge=with_answers)
    print_sweep(console, results)


if __name__ == "__main__":
    app()
