"""Cited answering.

The LLM is instructed to answer strictly from the retrieved context and to cite
sources with ``[n]`` markers. We then resolve every marker back to the actual
retrieved chunk it refers to and drop any marker that doesn't — so the returned
citations are guaranteed to point at real source spans, never fabricated ones.
"""

from __future__ import annotations

import re

from .llm import LLMClient
from .schemas import Answer, Citation, RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite every claim with the "
    "bracketed number of the passage it comes from, like [1] or [2][3]. "
    "If the context does not contain the answer, say you don't have enough "
    "information — do not use outside knowledge and do not invent citations."
)


def build_context(retrieval: RetrievalResult) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    blocks = []
    for rc in retrieval.retrieved:
        blocks.append(f"[{rc.rank}] (source: {rc.chunk.doc_id})\n{rc.chunk.text}")
    return "\n\n".join(blocks)


def _resolve_citations(text: str, retrieval: RetrievalResult) -> list[Citation]:
    """Map each ``[n]`` marker in the answer to its retrieved chunk.

    Markers that don't correspond to a retrieved rank are ignored (they would be
    fabricated sources). Returns one Citation per distinct valid marker, ordered
    by first appearance.
    """
    by_rank = {rc.rank: rc for rc in retrieval.retrieved}
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        marker = int(match.group(1))
        if marker in seen or marker not in by_rank:
            continue
        seen.add(marker)
        chunk = by_rank[marker].chunk
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
            )
        )
    return citations


def generate_answer(
    llm: LLMClient, question: str, retrieval: RetrievalResult
) -> Answer:
    """Produce a cited answer from the retrieved context."""
    context = build_context(retrieval)
    user_prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"
    text = llm.complete(SYSTEM_PROMPT, user_prompt).strip()
    citations = _resolve_citations(text, retrieval)
    return Answer(
        question=question,
        text=text,
        citations=citations,
        retrieval=retrieval,
        model=llm.name,
    )
