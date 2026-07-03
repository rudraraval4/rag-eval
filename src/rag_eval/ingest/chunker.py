"""Token-aware chunking with overlap.

Chunk quality is one of the biggest levers on RAG quality, so it's a
first-class, unit-tested component. We tokenize on a whitespace/punctuation
boundary (a fast, dependency-free approximation of model tokens) and slide a
fixed-size window with overlap across the token stream. Every chunk keeps its
exact character span in the source document, which is what makes citations
resolve to real text rather than paraphrases.

The overlap preserves context that would otherwise be split across a chunk
boundary (a definition and its use, a heading and its section).
"""

from __future__ import annotations

import re

from ..config import ChunkingConfig
from ..schemas import Chunk, Document

# A "token" here is a run of non-whitespace characters. This over-counts vs a
# BPE tokenizer but is stable, transparent, and good enough to size chunks.
_TOKEN_RE = re.compile(r"\S+")


def _tokens_with_spans(text: str) -> list[tuple[int, int]]:
    """Return (start_char, end_char) for every token in the text."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    """Split a document into overlapping chunks with exact char spans."""
    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if cfg.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    spans = _tokens_with_spans(doc.text)
    if not spans:
        return []

    step = cfg.chunk_size - cfg.chunk_overlap
    chunks: list[Chunk] = []
    chunk_index = 0

    for start_tok in range(0, len(spans), step):
        window = spans[start_tok : start_tok + cfg.chunk_size]
        if not window:
            break
        start_char = window[0][0]
        end_char = window[-1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{chunk_index}",
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                text=doc.text[start_char:end_char],
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata=dict(doc.metadata),
            )
        )
        chunk_index += 1

        # Reached the end of the document within this window; stop.
        if start_tok + cfg.chunk_size >= len(spans):
            break

    return chunks
