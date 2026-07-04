# Contributing

Thanks for your interest in `rag-eval`. This is a compact, well-tested codebase;
the guidelines below keep it that way.

## Development setup

```bash
python -m venv .venv
# activate: source .venv/bin/activate  (or .\.venv\Scripts\Activate.ps1 on Windows)
pip install -e ".[api,dev]"
cp .env.example .env          # add GROQ_API_KEY for the LLM stages
```

## Before you push

```bash
ruff check .      # lint (CI enforces this)
ruff format .     # auto-format
pytest -q         # tests must pass; LLMs are mocked, so no keys/network needed
```

CI runs ruff + pytest on every push and PR — keep both green.

## Architecture at a glance

- One `RunConfig` (Pydantic) threads through every stage; add a knob there.
- **Providers** live behind tiny protocols with factories: `embed.py`
  (`EmbeddingModel`) and `llm.py` (`LLMClient`). Add a provider by writing one
  class and registering it in the factory — nothing else changes.
- **Business logic is library code**; the CLI (`cli.py`) and API (`api.py`) are
  thin wiring, so everything is testable without them.
- The **eval harness** is under `eval/`: `retrieval_metrics.py` (deterministic),
  `judge.py` (LLM-as-judge), `runner.py` (concurrent), `sweep.py`, `artifacts.py`.

## Adding a provider

1. Implement the protocol method (`embed_documents`/`embed_query` or `complete`).
2. Lazy-import the SDK inside `__init__` and raise `MissingDependencyError` if
   it's absent; call `require_env(...)` for the API key.
3. Register it in the factory and add it to the `pyproject.toml` extras.
4. Add a factory test (see `tests/test_provider_factory.py`).

## Tests

Every new behavior needs a test. Keep LLMs mocked (`MockLLM`) so the suite runs
offline and deterministically.
