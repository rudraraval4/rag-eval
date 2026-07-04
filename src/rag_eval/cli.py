"""Command-line interface: ingest, ask, eval, sweep, doctor.

Thin wiring only — each command loads a RunConfig, applies CLI overrides, and
delegates to the pipeline modules. Business logic lives in the library so it
stays testable without the CLI. Progress bars and spinners give a non-technical
user clear feedback that work is happening and roughly how long it will take.
"""

from __future__ import annotations

import logging
import os

# Quiet the noisy ML-stack warnings before anything imports them.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path  # noqa: E402

import typer  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402

from .config import RunConfig  # noqa: E402
from .observability import configure_logging  # noqa: E402

app = typer.Typer(
    add_completion=False,
    help="A production-grade RAG pipeline with a first-class evaluation harness.",
)
console = Console()

DEFAULT_CONFIG = "configs/default.yaml"


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only."),
) -> None:
    """Set up logging for every command."""
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    configure_logging(level=level)


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
def doctor(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
) -> None:
    """Check that everything is set up correctly before running."""
    cfg = _load_config(config)
    from .diagnostics import run_diagnostics
    from .render import print_doctor

    ok = print_doctor(console, run_diagnostics(cfg))
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def ingest(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
    corpus_dir: str | None = typer.Option(None, help="Override the corpus directory."),
    rebuild: bool = typer.Option(False, help="Drop and rebuild the collection from scratch."),
) -> None:
    """Build the vector index from a folder of documents."""
    cfg = _load_config(config, **{"paths.corpus_dir": corpus_dir})
    from .pipeline import run_ingest

    with console.status("[bold]Loading, chunking, and embedding documents…[/]"):
        stats = run_ingest(cfg, rebuild=rebuild)
    console.print(
        f"[green]✓ Ingested[/] {stats['documents']} documents "
        f"→ {stats['chunks']} chunks into '{cfg.storage.collection}'."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
    top_k: int | None = typer.Option(None, help="Override number of chunks to retrieve."),
    llm_provider: str | None = typer.Option(None, help="Override LLM provider."),
    llm_model: str | None = typer.Option(None, help="Override LLM model."),
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

    with console.status("[bold]Retrieving context and generating an answer…[/]"):
        answer = run_ask(cfg, question)
    print_answer(console, answer)


@app.command()
def eval(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to the run config YAML."),
    no_judge: bool = typer.Option(False, help="Skip LLM-as-judge; retrieval metrics only."),
    limit: int | None = typer.Option(None, help="Only evaluate the first N cases."),
) -> None:
    """Run the evaluation suite and print a scorecard."""
    cfg = _load_config(config)
    from .eval.runner import run_eval
    from .render import print_scorecard, progress_callback

    console.print(
        "[dim]Evaluating — retrieval is instant; answering + judging run "
        f"{cfg.eval.concurrency} at a time. This can take a minute.[/]"
    )
    with progress_callback(console, "Scoring cases") as on_progress:
        scorecard = run_eval(
            cfg, judge=not no_judge, limit=limit, on_progress=on_progress
        )
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

    with console.status("[bold]Re-indexing and evaluating each configuration…[/]"):
        results = run_sweep(cfg, answers=with_answers, judge=with_answers)
    print_sweep(console, results)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev)."),
) -> None:
    """Run the REST API server (requires the 'api' extra)."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]The API server needs the 'api' extra:[/] pip install -e \".[api]\""
        )
        raise typer.Exit(code=1) from None
    console.print(f"[green]Serving rag-eval API at[/] http://{host}:{port}  (docs at /docs)")
    uvicorn.run("rag_eval.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
