<div align="center">

# 🔍 rag-eval

**A production-grade RAG system whose real feature is _measuring itself_.**

[![CI](https://github.com/rudraraval4/rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/rudraraval4/rag-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen.svg)](tests)

<img src="docs/demo.svg" alt="rag-eval terminal demo: a readiness check, a cited answer, an honest decline, and the config-comparison table" width="780">

</div>

Everyone builds "chat with your PDF." Almost nobody proves theirs is any good.
`rag-eval` ships a clean retrieval-augmented-generation pipeline **and** a
first-class evaluation harness — deterministic retrieval metrics plus an
LLM-as-judge scorecard — so every configuration change is backed by numbers,
not vibes.

### Why it reads as production-grade

| | |
|---|---|
| 📊 **Measured, not vibes** | recall@k · hit-rate · MRR · precision@k, plus LLM-judge faithfulness / relevance / hallucination |
| 🔌 **Pluggable providers** | Embeddings: BGE · OpenAI · Voyage — LLM: Groq · OpenAI · Anthropic. One flag to swap. |
| 🧾 **Real citations** | every `[n]` resolves to an actual retrieved chunk; fabricated markers are dropped |
| ⚡ **Fast & resilient** | concurrent eval · retry + backoff on rate-limits · progress bars with ETA |
| 🩺 **Operable** | `doctor` preflight · structured logs · saved run artifacts · config-driven |
| 🚀 **Three surfaces** | CLI · importable Python library · REST API (FastAPI + Docker) · CI |

---

## The scorecard (real output)

`rag-eval eval` on the committed corpus (10 documents) and labeled eval set
(25 questions), default config — local BGE embeddings, Groq
`llama-3.3-70b-versatile` for answering and judging:

```
Retrieval metrics (k=5, n=25)        Answer-quality (LLM-as-judge)
┌─────────────┬───────┐              ┌────────────────────┬───────┐
│ hit_rate    │ 1.000 │              │ faithfulness       │ 1.000 │
│ recall@5    │ 1.000 │              │ answer_relevance   │ 0.992 │
│ precision@5 │ 0.217 │              │ hallucination_rate │ 0.000 │
│ MRR         │ 0.893 │              └────────────────────┴───────┘
└─────────────┴───────┘
```

- **Retrieval metrics are deterministic** — no LLM, no flakiness. `recall@5 =
  1.0` means the right document was always in the top 5; `MRR = 0.893` means it
  was usually rank 1. `precision@5 = 0.217` is expected and honest: most
  questions have one relevant document, so at most 1 of 5 retrieved chunks is
  "relevant."
- **Answer quality is LLM-judged** at temperature 0 with pinned prompts.
  `hallucination_rate = 0.0` includes the two deliberately **unanswerable**
  questions — the system correctly declines instead of fabricating.

Every run is saved to `runs/<timestamp>/scorecard.{json,md}` — auditable and
diffable.

## Which configuration is best? (the experiment)

`rag-eval sweep` re-indexes across chunk sizes and sweeps `k`, reporting the
deterministic retrieval metrics chunking actually moves. One command, one table:

| chunk | k | recall@k | precision@k | MRR |
|------:|--:|---------:|------------:|------:|
| 64  | 3 | 0.928 | **0.652** | 0.928 |
| 64  | 5 | 0.986 | 0.461 | 0.938 |
| **128** | **3** | 0.942 | 0.551 | **0.957** |
| 128 | 5 | 0.957 | 0.357 | 0.957 |
| 256 | 3 | 0.957 | 0.348 | 0.884 |
| 256 | 5 | **1.000** | 0.217 | 0.893 |

**Finding:** for this corpus of short, single-topic documents, **chunk size 128
gives the best ranking (MRR 0.957)**. Tiny chunks (64) maximize precision but
fragment context; whole-document chunks (256) maximize `recall@5` but rank the
right document lower. A decision teams usually make by gut — here it's measured.

And because providers are swappable, the harness catches quality regressions
from a weaker model — same retrieval, measurably weaker answers:

| answer model | recall@5 | MRR | faithfulness | answer_relevance |
|---|---:|---:|---:|---:|
| Groq `llama-3.3-70b` | 1.000 | 0.893 | **1.000** | **0.992** |
| Groq `llama-3.1-8b`  | 1.000 | 0.893 | 0.960 | 0.860 |

## Honest failure the eval catches

The eval set includes questions the corpus **cannot** answer (e.g. "configure
Git to sign commits with an SSH key"). A naive RAG system hallucinates an
answer. Here the answerer declines and the judge confirms it — those cases score
faithful and non-hallucinated *because the system refused*. The harness lets you
assert that, instead of hoping.

---

## Pluggable providers

Swap embedding and LLM providers via config or CLI flags. Paid providers
activate only when you supply an API key (lazy-imported optional extras keep the
base install lean).

| Layer | Default (no extra key) | Opt-in |
|---|---|---|
| Embeddings | **BGE** (`bge-small-en-v1.5`, local, offline) | OpenAI, Voyage |
| LLM (answer + judge) | **Groq** (`llama-3.3-70b-versatile`) | OpenAI, Anthropic (Claude) |

```bash
# Answer with Claude, keep Groq as an independent judge — no code change
rag-eval ask "..." --llm-provider anthropic --llm-model claude-sonnet-5
```

The judge is configured separately from the answerer, so you can point it at a
different (or stronger) model to reduce self-preference bias.

## Quickstart

```bash
python -m venv .venv
# activate: source .venv/bin/activate        (macOS/Linux)
#           .\.venv\Scripts\Activate.ps1     (Windows PowerShell)
pip install -e ".[api,dev]"
cp .env.example .env            # add GROQ_API_KEY (free at console.groq.com/keys)

rag-eval doctor                 # preflight: keys, corpus, eval set, index
rag-eval ingest                 # build the vector index from data/corpus
rag-eval ask "How do I create a branch and switch to it?"
rag-eval eval                   # scorecard (progress bar + ETA, saved to runs/)
rag-eval sweep                  # config-comparison table
```

> The `rag-eval` command lives inside the virtualenv — activate it first, or
> call it directly with `.venv/bin/rag-eval` / `.\.venv\Scripts\rag-eval.exe`.
> Every command takes `--config configs/groq-8b.yaml` for a smaller/faster model.

## Use your own data

1. Drop your files (`.md`, `.txt`, `.pdf`, `.html`) into `data/corpus/`.
2. `rag-eval ingest --rebuild` → now it answers over your documents.
3. To evaluate on your data, write a `data/eval/eval_set.jsonl` — one JSON line
   per case: `{"case_id", "question", "ideal_answer", "relevant_doc_ids": [...]}`.
   That labeled set is what turns "seems fine" into "recall@5 = 0.94."

Everything (folders, providers, chunk size, `k`, concurrency) lives in
`configs/*.yaml` — no code changes.

## REST API

Run it as a service:

```bash
pip install -e ".[api]"
rag-eval serve                  # http://127.0.0.1:8000  (interactive docs at /docs)
```

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | liveness |
| GET  | `/config` | active configuration summary |
| POST | `/ingest` | build/rebuild the index (`{"rebuild": true}`) |
| POST | `/ask`    | `{"question": "...", "top_k": 5}` → answer + resolved citations |
| POST | `/eval`   | `{"judge": true, "limit": 25}` → scorecard summary |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is the difference between git reset --hard and git revert?"}'
```

### Docker

```bash
docker compose up --build       # API on :8000, index persisted in a volume
# then: curl -X POST localhost:8000/ingest -d '{}' -H 'content-type: application/json'
```

Local BGE embeddings run on CPU — no GPU required. Provide `GROQ_API_KEY` via
`.env` (compose reads it automatically).

## Production features

- **Retry with backoff** — every provider call is wrapped; transient 429s / 5xx
  are retried with jittered exponential backoff, non-transient errors fail fast.
- **Concurrent evaluation** — answer + judge calls run across a bounded thread
  pool (`eval.concurrency`), turning a minutes-long eval into seconds.
- **Progress + feedback** — `eval` shows a live progress bar with ETA; `ingest`
  and `ask` show spinners; ML-stack warning noise is suppressed.
- **Structured logging** — console (`--verbose`/`--quiet`) and a rotating file
  under `logs/`, capturing provider latency, token usage, and retries.
- **Run artifacts** — every eval writes `runs/<timestamp>/scorecard.{json,md}`.
- **`doctor`** — one command validates keys, corpus, eval set, and index before
  you spend time on a run that would fail.

## Architecture

```
 docs/ (md, txt, pdf, html)                          ┌──────────────┐
        │  INGEST                                     │  Chroma      │
        └─▶ load → chunk (token-aware, overlap) ──────┤  (persistent)│
              → embed (pluggable: BGE/OpenAI/Voyage)  └──────┬───────┘
                                                             │
 question ─▶ embed query → retrieve top-k ───────────────────┘
              → numbered context → LLM answers with [n] citations
              → resolve every [n] to a real chunk (drop fabricated ones)
                                    │
                                    ▼  answer + verified citations

 eval-set ─▶ RETRIEVAL METRICS (deterministic): recall@k, hit-rate, MRR, precision@k
          ─▶ ANSWER METRICS (LLM-as-judge): faithfulness, relevance, hallucination
          ─▶ SCORECARD (saved) + CONFIG SWEEP → comparison table

 surfaces:  CLI (typer)  ·  Python library  ·  REST API (FastAPI + Docker)
```

One `RunConfig` (Pydantic) threads through every stage, so a change to chunk
size, `k`, or provider is a single edit and every run is reproducible.

## Project layout

```
src/rag_eval/
  config.py schemas.py errors.py       # typed core + friendly errors
  observability.py resilience.py       # logging + retry/backoff
  embed.py  llm.py                     # pluggable providers (factory + lazy imports)
  ingest/{loaders,chunker}.py          # md/txt/pdf/html loaders + token-aware chunker
  store.py retrieve.py answer.py       # Chroma + retrieval + cited answering
  pipeline.py render.py cli.py         # wiring, rich output, Typer CLI
  diagnostics.py api.py                # doctor checks, FastAPI service
  eval/
    retrieval_metrics.py judge.py      # deterministic metrics + LLM-as-judge
    dataset.py runner.py sweep.py artifacts.py
configs/                               # default.yaml, groq-8b.yaml
data/corpus/  data/eval/eval_set.jsonl # committed sample corpus + 25 labeled cases
tests/                                 # 39 tests (pipeline + eval + API + retry)
Dockerfile docker-compose.yml .github/workflows/ci.yml
```

## Testing & CI

```bash
pytest        # 39 tests: chunking, retrieval metrics, citation resolution,
              # judge parsing, provider selection, retry, run artifacts, REST API.
              # LLMs are mocked — no network, no keys.
ruff check .  # lint (enforced in CI)
```

GitHub Actions runs ruff + pytest on every push and PR.

## Limitations (honest notes)

- The corpus (10 Git-concept docs) and eval set (25 questions) are a **small,
  illustrative sample** chosen so ground truth is unambiguous and reproducible
  offline — they demonstrate the harness, they are not a general benchmark.
  Point the config at your own corpus and eval set for numbers that mean
  something for your data.
- The default judge shares a model family with the answerer; configure a
  different `judge.provider`/`judge.model` for a more independent signal.
- The chunker approximates tokens on a whitespace boundary — transparent and
  dependency-free, but not identical to a model's BPE tokenizer.

## Alternatives considered

The evaluator is hand-rolled on purpose — owning the retrieval metrics and judge
prompts is the point. [ragas](https://github.com/explodinggradients/ragas) is a
capable off-the-shelf alternative if you'd rather not maintain your own.

## License

MIT — see [LICENSE](LICENSE).
