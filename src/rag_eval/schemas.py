"""Typed data models that flow through the pipeline.

Everything the pipeline produces is a validated Pydantic model so that
ingestion, retrieval, answering, and evaluation all speak the same language and
serialize cleanly to JSON (for cached indexes, scorecards, and run artifacts).
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class Document(BaseModel):
    """A single source document after loading and cleaning."""

    doc_id: str = Field(..., description="Stable id, typically derived from the source path.")
    source_path: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A contiguous slice of a document, the unit that gets embedded and stored."""

    chunk_id: str = Field(..., description="Globally unique: '{doc_id}::{chunk_index}'.")
    doc_id: str
    source_path: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """A chunk returned by the vector store with its similarity score."""

    chunk: Chunk
    score: float = Field(..., description="Similarity score; higher is more relevant.")
    rank: int = Field(..., description="1-based position in the retrieved list.")


class RetrievalResult(BaseModel):
    """The ordered top-k chunks retrieved for a query."""

    query: str
    retrieved: list[RetrievedChunk] = Field(default_factory=list)

    @property
    def doc_ids(self) -> list[str]:
        """Distinct document ids present in the retrieved chunks, in rank order."""
        seen: list[str] = []
        for rc in self.retrieved:
            if rc.chunk.doc_id not in seen:
                seen.append(rc.chunk.doc_id)
        return seen


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A resolved inline citation marker [n] pointing at a real retrieved chunk."""

    marker: int = Field(..., description="The [n] marker used in the answer text.")
    chunk_id: str
    doc_id: str
    source_path: str


class Answer(BaseModel):
    """An LLM answer with citations that resolve to actual retrieved chunks."""

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval: RetrievalResult
    model: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class EvalCase(BaseModel):
    """One labeled example from the eval set."""

    case_id: str
    question: str
    ideal_answer: str = ""
    relevant_doc_ids: list[str] = Field(
        default_factory=list,
        description="Ground-truth documents that should be retrieved for this question.",
    )
    unanswerable: bool = Field(
        default=False,
        description="If true, a good system should decline rather than fabricate.",
    )


class RetrievalMetrics(BaseModel):
    """Deterministic retrieval metrics for a single query or aggregated over a set."""

    hit_rate: float = Field(..., description="1.0 if any relevant doc was retrieved, else 0.")
    recall_at_k: float
    precision_at_k: float
    mrr: float = Field(..., description="Reciprocal rank of the first relevant doc.")
    k: int


class AnswerJudgement(BaseModel):
    """LLM-as-judge scores for a single answer. Scores are in [0, 1]."""

    faithfulness: float = Field(..., description="Is the answer grounded in retrieved context?")
    answer_relevance: float = Field(..., description="Does the answer address the question?")
    hallucination: bool = Field(..., description="True if the answer asserts unsupported claims.")
    rationale: str = ""


class CaseResult(BaseModel):
    """Full evaluation outcome for one eval case."""

    case: EvalCase
    retrieval_metrics: RetrievalMetrics
    answer: Answer | None = None
    judgement: AnswerJudgement | None = None


class Scorecard(BaseModel):
    """Aggregated results of an evaluation run over the whole eval set."""

    config_summary: dict[str, str] = Field(default_factory=dict)
    num_cases: int = 0
    # Aggregated retrieval metrics
    hit_rate: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    k: int = 0
    # Aggregated answer metrics (None when judging was skipped)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    hallucination_rate: float | None = None
    # Detail
    results: list[CaseResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
