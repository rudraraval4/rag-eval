# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Pluggable providers**, hardened: BGE (local, default), OpenAI, and Voyage
  embeddings; Groq (default), OpenAI, and Anthropic (Claude) LLMs — selectable
  via config or CLI flags.
- **Retry with exponential backoff** around every provider call; transient
  rate-limits and 5xx are retried, non-transient errors fail fast.
- **Concurrent evaluation** — answer and judge calls run across a bounded thread
  pool, cutting a 25-case eval from minutes to seconds.
- **Progress bars** (with ETA) on `eval`, spinners on `ingest`/`ask`, and quiet
  suppression of ML-stack warning noise.
- **Structured logging** to console and a rotating file under `logs/`.
- **Run artifacts** — every eval writes `runs/<timestamp>/scorecard.{json,md}`.
- **`doctor`** command — preflight checks for keys, corpus, eval set, and index.
- **REST API** (FastAPI): `/health`, `/config`, `/ingest`, `/ask`, `/eval`, plus
  a `serve` command and a Docker image.
- **CI** (GitHub Actions): ruff + pytest on every push and PR.

### Notes
- Retrieval + ingest are fully offline (local BGE); answering/eval need one free
  Groq key by default.

## [0.1.0] — Initial

### Added
- RAG pipeline: ingest (md/txt/pdf/html) → token-aware chunking → embed → Chroma
  → retrieve → cited answers with citations resolved to real chunks.
- Evaluation harness: deterministic retrieval metrics (recall@k, hit-rate, MRR,
  precision@k) + LLM-as-judge (faithfulness, relevance, hallucination).
- Config-driven pipeline, config sweep, committed corpus + labeled eval set,
  and a test suite.
