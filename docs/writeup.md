# Stop shipping un-evaluated RAG — here's how I measured mine

RAG is easy to demo and hard to trust. You wire up an embedding model, a vector
store, and an LLM; you ask it three questions that happen to work; you ship. The
problem shows up later, silently: someone bumps the chunk size, swaps the
embedding model, or changes `k`, the answers quietly get worse, and nobody
notices because there was never a number to watch.

I built [`rag-eval`](https://github.com/rudraraval4/rag-eval) to make that
failure mode impossible. It's a normal RAG pipeline — ingest, chunk, embed,
retrieve, answer with citations. The part that matters is that it grades itself.

## Two kinds of "is it good?"

RAG has two failure surfaces, and they need different tools.

**Did we retrieve the right thing?** This is a pure information-retrieval
question, and you can answer it with exact math — no LLM required. Given a
labeled set of questions with their ground-truth documents, you compute:

- **hit-rate** — did any relevant document make the top-k?
- **recall@k** — what fraction of relevant documents did we retrieve?
- **precision@k** — what fraction of what we retrieved was relevant?
- **MRR** — how high did the first relevant document rank?

These are deterministic and flake-free. This is the credible half of the
scorecard, and most "chat with your docs" projects skip it entirely.

**Was the answer any good?** This one genuinely needs a model. I use an
LLM-as-judge at temperature 0 with pinned prompts to score three things:
faithfulness (is every claim grounded in the retrieved context?), answer
relevance, and a hallucination flag. Crucially, the judge is configured
separately from the answerer — you can point it at a different model, because a
model grading its own output has a known self-preference bias.

## What measurement actually buys you

Here's the config-comparison table `rag-eval sweep` produces, sweeping chunk size
and `k` over a small Git-documentation corpus:

| chunk | k | recall@k | precision@k | MRR |
|------:|--:|---------:|------------:|------:|
| 64  | 3 | 0.928 | **0.652** | 0.928 |
| 128 | 3 | 0.942 | 0.551 | **0.957** |
| 256 | 5 | **1.000** | 0.217 | 0.893 |

Every row is a decision someone would otherwise make by feel. Tiny chunks (64)
give the best precision but fragment context. Whole-document chunks (256) get the
right doc into the top-5 every time, but rank it lower and bury it among
irrelevant neighbours. Mid-size chunks (128) win on ranking. "It depends" is a
real answer — but now it depends on a number instead of a hunch.

## The test that separates real systems from demos

The eval set includes questions the corpus **cannot** answer. A demo-grade RAG
system will confidently make something up. A real one declines — and you should
be able to prove it declines, not just hope so. In `rag-eval` those cases score
faithful and non-hallucinated precisely *because* the system refused, and the
judge confirms it. That single property — "does it know what it doesn't know?" —
is invisible without an eval harness.

## Two implementation details that matter

**Citations that resolve to real chunks.** The answerer cites with `[n]` markers.
After generation, every marker is matched back to the exact retrieved chunk it
refers to, and any marker with no corresponding retrieval is dropped as
fabricated. A citation you can't trace isn't a citation.

**Config-driven everything.** One typed config object threads through ingestion,
retrieval, answering, and evaluation. That's what makes the sweep a single
command instead of a weekend of manual runs — and what makes every number in the
scorecard reproducible from a committed corpus and eval set.

## The uncomfortable takeaway

The eval harness was maybe a third of the code, and it's the only part that tells
me whether the other two-thirds work. If you're building RAG and you can't answer
"what's your recall@k?" and "what's your hallucination rate?", you don't have a
retrieval system — you have a demo that hasn't failed *yet*.

Measuring it is not that much work. Here's [the whole
thing](https://github.com/rudraraval4/rag-eval).
