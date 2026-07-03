"""Configuration sweep — the experiment that proves which setup is best.

Runs the full eval across a grid of chunk sizes and k values and returns one
scorecard per configuration, ready to render as a comparison table. Chunk-size
changes require re-indexing, so we rebuild the store once per chunking config
and reuse it across the k values that share it (cheap, since only retrieval
depth changes).
"""

from __future__ import annotations

from ..config import RunConfig
from ..pipeline import run_ingest
from ..schemas import Scorecard
from .runner import run_eval

# The grid to explore. Chunk sizes are chosen to actually fragment this
# corpus's short docs (each ~140-180 tokens): 256 ≈ one chunk per doc, 128 ≈
# two, 64 ≈ three — so the table shows how over-chunking trades off against
# whole-document retrieval. Kept small so a portfolio demo runs in seconds.
CHUNKING_GRID: list[tuple[int, int]] = [(64, 8), (128, 16), (256, 32)]
TOP_K_GRID: list[int] = [3, 5]


def run_sweep(base: RunConfig, *, answers: bool = False, judge: bool = False) -> list[Scorecard]:
    """Sweep chunk_size × top_k and return a scorecard per configuration.

    Retrieval-only by default (``answers=False``): the deterministic retrieval
    metrics are exactly what chunk size and k affect, so the comparison is fast,
    reproducible, and needs no LLM calls. Pass ``answers=True`` to also score
    answer quality per configuration (many more LLM calls).
    """
    results: list[Scorecard] = []
    for chunk_size, overlap in CHUNKING_GRID:
        chunk_cfg = base.with_overrides(
            **{"chunking.chunk_size": chunk_size, "chunking.chunk_overlap": overlap}
        )
        # Re-index once for this chunking configuration.
        run_ingest(chunk_cfg, rebuild=True)
        for top_k in TOP_K_GRID:
            run_cfg = chunk_cfg.with_overrides(**{"retrieval.top_k": top_k})
            results.append(run_eval(run_cfg, answers=answers, judge=judge))
    return results
