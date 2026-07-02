# rag-eval

**A production-grade RAG system whose real feature is measuring itself.**

Everyone builds "chat with your PDF." Almost nobody proves theirs is any good.
`rag-eval` ships a clean retrieval-augmented-generation pipeline **and** a
first-class evaluation harness — deterministic retrieval metrics plus an
LLM-as-judge scorecard — so every configuration change is backed by numbers,
not vibes.

> 🚧 Work in progress. This README will grow into the full quickstart, eval
> scorecard, and config-comparison table as the build lands.

## Why this exists

RAG quality silently degrades with the wrong chunk size, `k`, or embedding
model — and you'd never know without measurement. This project treats
evaluation as a first-class citizen:

- **Retrieval metrics** (deterministic, no LLM): recall@k, hit-rate, MRR,
  precision@k — did we fetch the right chunks?
- **Answer-quality metrics** (LLM-as-judge): faithfulness, answer relevance,
  and hallucination flags — is the answer actually grounded in what we
  retrieved?
- **Citations that resolve to real retrieved chunks** — verified, never
  fabricated.
- **Config-driven experiments** and a comparison table that proves which setup
  wins.

## Pluggable providers

Swap embedding and LLM providers via config; paid providers activate only when
you supply an API key.

| Layer | Default (no extra key) | Opt-in |
|---|---|---|
| Embeddings | **BGE** (local, offline) | OpenAI, Voyage |
| LLM (answer + judge) | **Groq** (`llama-3.3-70b-versatile`) | OpenAI, Anthropic (Claude) |

Ingest + retrieval are fully keyless (local BGE). Answering and evaluation need
one free [Groq](https://console.groq.com/keys) key by default.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env          # add GROQ_API_KEY
rag-eval ingest               # build the vector index from data/corpus
rag-eval ask "How do I create a branch and switch to it?"
rag-eval eval                 # run the scorecard
```

## License

MIT — see [LICENSE](LICENSE).
