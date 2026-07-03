# rag-eval

**A production-grade RAG system whose real feature is measuring itself.**

Everyone builds "chat with your PDF." Almost nobody proves theirs is any good.
`rag-eval` ships a clean retrieval-augmented-generation pipeline **and** a
first-class evaluation harness — deterministic retrieval metrics plus an
LLM-as-judge scorecard — so every configuration change is backed by numbers,
not vibes.

The pipeline is deliberately standard. **The eval is the point.**

---

## The scorecard (real output)

`rag-eval eval` on the committed corpus (10 documents) and labeled eval set
(25 questions), default config — local BGE embeddings, Groq
`llama-3.3-70b-versatile` for both answering and judging:

```
Run configuration
  embedding=bge:BAAI/bge-small-en-v1.5 · llm=groq:llama-3.3-70b-versatile
  judge=groq:llama-3.3-70b-versatile · chunk_size=512 · overlap=64 · top_k=5

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
  questions have a single relevant document, so at most 1 of 5 retrieved chunks
  can be "relevant."
- **Answer quality is LLM-judged** at temperature 0 with pinned prompts:
  faithfulness (is every claim grounded in retrieved context?), answer relevance,
  and a hallucination flag. `hallucination_rate = 0.0` includes the two
  deliberately **unanswerable** questions — the system correctly declines
  instead of fabricating.

## Which configuration is best? (the experiment)

`rag-eval sweep` re-indexes across chunk sizes and sweeps `k`, reporting the
deterministic retrieval metrics that chunking actually moves. One command, one
table:

| chunk | overlap | k | hit_rate | recall@k | precision@k | MRR |
|------:|--------:|--:|---------:|---------:|------------:|------:|
| 64  | 8  | 3 | 0.957 | 0.928 | **0.652** | 0.928 |
| 64  | 8  | 5 | 1.000 | 0.986 | 0.461 | 0.938 |
| **128** | **16** | **3** | 0.957 | 0.942 | 0.551 | **0.957** |
| 128 | 16 | 5 | 0.957 | 0.957 | 0.357 | 0.957 |
| 256 | 32 | 3 | 0.957 | 0.957 | 0.348 | 0.884 |
| 256 | 32 | 5 | 1.000 | **1.000** | 0.217 | 0.893 |

**Finding:** for this corpus of short, single-topic documents, **chunk size 128
gives the best ranking (MRR 0.957)** and a strong recall/precision balance. Tiny
chunks (64) maximize precision but fragment context; whole-document chunks (256)
maximize `recall@5` but rank the right document lower and drown it among
irrelevant neighbours. This is the kind of decision teams usually make by gut —
here it's a measurement.

## Honest failure the eval catches

The eval set includes questions the corpus **cannot** answer (e.g. "How do I
configure Git to sign commits with an SSH key?"). A naive RAG system happily
hallucinates an answer. Here, retrieval still returns the nearest chunks, but the
answerer is instructed to decline, and the judge confirms it: those cases score
faithful and non-hallucinated **because the system refused**. The harness is
what lets you assert that — instead of hoping.

---

## Architecture

```
 docs/ (md, txt, pdf, html)                          ┌──────────────┐
        │  INGEST                                     │  Chroma      │
        └─▶ load → chunk (token-aware, overlap) ──────┤  (persistent)│
              → embed (BGE, local, no key)            └──────┬───────┘
                                                             │
 question ─▶ embed query → retrieve top-k ───────────────────┘
              → build numbered context
              → LLM answers with [n] citations
              → resolve every [n] to a real chunk (drop fabricated ones)
                                    │
                                    ▼   answer + verified citations

 eval-set ─▶ RETRIEVAL METRICS (deterministic): recall@k, hit-rate, MRR, precision@k
          ─▶ ANSWER METRICS (LLM-as-judge): faithfulness, relevance, hallucination
          ─▶ SCORECARD  +  CONFIG SWEEP (chunk size × k) → comparison table
```

One `RunConfig` (Pydantic) threads through every stage, so a change to chunk
size, `k`, or provider is a single edit and every run is reproducible.

## Pluggable providers

Swap embedding and LLM providers via config or CLI flags; paid providers activate
only when you supply an API key (lazy-imported optional extras keep the base
install lean).

| Layer | Default (no extra key) | Opt-in |
|---|---|---|
| Embeddings | **BGE** (`bge-small-en-v1.5`, local, offline) | OpenAI, Voyage |
| LLM (answer + judge) | **Groq** (`llama-3.3-70b-versatile`) | OpenAI, Anthropic (Claude) |

Ingest + retrieval are fully keyless (local BGE). Answering and evaluation need
one free [Groq](https://console.groq.com/keys) key by default. The judge is
configured separately from the answerer, so you can point it at a different (or
stronger) model to reduce self-preference bias.

```bash
# use Claude to answer, keep Groq as the judge — no code change
rag-eval ask "..." --llm-provider anthropic --llm-model claude-sonnet-5
```

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env            # add GROQ_API_KEY (free at console.groq.com/keys)

rag-eval ingest                 # build the vector index from data/corpus
rag-eval ask "How do I create a branch and switch to it?"
rag-eval eval                   # print the scorecard
rag-eval sweep                  # print the config-comparison table
```

Example `ask` output:

```
╭───────────────────────────── Answer ─────────────────────────────╮
│ You can create a branch and switch to it in one step using        │
│ `git switch -c <name>` [1] or the older `git checkout -b          │
│ <name>` [1].                                                      │
╰───────────────────────────────────────────────────────────────────╯
  Citations
  # │ Source    │ Chunk
  1 │ branching │ branching::0
  model: groq:llama-3.3-70b-versatile · retrieved: 5 chunks
```

## Project layout

```
src/rag_eval/
  config.py schemas.py errors.py   # typed core (Pydantic) + friendly errors
  embed.py  llm.py                 # pluggable providers (factory + lazy imports)
  ingest/{loaders,chunker}.py      # md/txt/pdf/html loaders + token-aware chunker
  store.py retrieve.py answer.py   # Chroma + retrieval + cited answering
  pipeline.py render.py cli.py     # wiring, rich output, Typer CLI
  eval/
    retrieval_metrics.py           # recall@k, hit-rate, MRR, precision@k
    judge.py                       # faithfulness / relevance / hallucination
    dataset.py runner.py sweep.py  # eval set, scorecard, config comparison
data/corpus/                       # committed sample documents
data/eval/eval_set.jsonl           # 25 labeled cases (2 unanswerable)
tests/                             # 25 tests
```

## Testing

```bash
pytest        # 25 tests: chunking spans/overlap, retrieval metrics, citation
              # resolution (incl. dropping fabricated cites), judge parsing,
              # provider selection. LLMs are mocked — no network, no keys.
```

## Design choices worth calling out

- **Citations resolve to real retrieved chunks.** The answer's `[n]` markers are
  matched back to the exact chunk they reference; any marker with no
  corresponding retrieval is dropped as fabricated. No invented sources.
- **The eval set ships with the repo.** Corpus + labeled questions are committed,
  so anyone can reproduce every number above.
- **Retrieval eval is LLM-free.** The credible half of the scorecard is exact and
  deterministic; only answer quality uses a judge.

## Limitations (honest notes)

- The corpus (10 Git-concept docs) and eval set (25 questions) are a **small,
  illustrative sample** chosen so ground truth is unambiguous and fully
  reproducible offline — the metrics demonstrate the harness, they are not a
  general RAG benchmark. Point the config at your own corpus and eval set to get
  numbers that mean something for your data.
- The default judge shares a model family with the answerer. LLM-as-judge has
  known self-preference bias; configure a different `judge.provider`/`judge.model`
  for a more independent signal (the plumbing is already there).
- The chunker approximates tokens on a whitespace boundary — transparent and
  dependency-free, but not identical to a model's BPE tokenizer.

## Alternatives considered

The evaluator is hand-rolled on purpose — owning the retrieval metrics and judge
prompts is the whole point of the project. [ragas](https://github.com/explodinggradients/ragas)
is a capable off-the-shelf alternative if you'd rather not maintain your own.

## License

MIT — see [LICENSE](LICENSE).
