"""FastAPI service — run the pipeline as an HTTP backend.

Exposes ingest / ask / eval over REST so a company can deploy `rag-eval` as a
service instead of a CLI. Heavy components (the embedding model, the vector
store) are built once and reused across requests. Install with the ``api``
extra: ``pip install -e ".[api]"``, then ``rag-eval serve`` or
``uvicorn rag_eval.api:app``.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .answer import generate_answer
from .config import RunConfig
from .llm import get_llm_client
from .observability import configure_logging, get_logger
from .pipeline import build_retriever, run_ingest

_log = get_logger("api")


class _Service:
    """Holds config and lazily-built, reused pipeline components."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self._retriever = None
        self._llm = None

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = build_retriever(self.cfg)
        return self._retriever

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_client(self.cfg.llm, self.cfg.retry)
        return self._llm

    def invalidate(self) -> None:
        self._retriever = None  # index changed; rebuild on next use


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1)


class Source(BaseModel):
    marker: int
    doc_id: str
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Source]
    model: str
    retrieved: int


class IngestRequest(BaseModel):
    rebuild: bool = False


class EvalRequest(BaseModel):
    judge: bool = True
    limit: int | None = Field(None, ge=1)


def create_app(cfg: RunConfig | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a config for tests; defaults to YAML/env."""
    load_dotenv()
    configure_logging()
    if cfg is None:
        default_path = Path("configs/default.yaml")
        cfg = RunConfig.from_yaml(default_path) if default_path.exists() else RunConfig()
    service = _Service(cfg)

    app = FastAPI(
        title="rag-eval",
        version="0.1.0",
        description="Retrieval-augmented generation with a first-class eval harness.",
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def config() -> dict[str, str]:
        return service.cfg.summary()

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, int]:
        stats = run_ingest(service.cfg, rebuild=req.rebuild)
        service.invalidate()
        _log.info("Ingested %s", stats)
        return stats

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        retrieval = service.retriever.retrieve(req.question, req.top_k)
        answer = generate_answer(service.llm, req.question, retrieval)
        by_id = {rc.chunk.chunk_id: rc.score for rc in retrieval.retrieved}
        return AskResponse(
            question=answer.question,
            answer=answer.text,
            citations=[
                Source(
                    marker=c.marker,
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    score=by_id.get(c.chunk_id, 0.0),
                )
                for c in answer.citations
            ],
            model=answer.model,
            retrieved=len(retrieval.retrieved),
        )

    @app.post("/eval")
    def evaluate(req: EvalRequest) -> dict[str, object]:
        from .eval.runner import run_eval

        sc = run_eval(service.cfg, judge=req.judge, limit=req.limit, save=False)
        return {
            "num_cases": sc.num_cases,
            "k": sc.k,
            "retrieval": {
                "hit_rate": sc.hit_rate,
                "recall_at_k": sc.recall_at_k,
                "precision_at_k": sc.precision_at_k,
                "mrr": sc.mrr,
            },
            "answer_quality": {
                "faithfulness": sc.faithfulness,
                "answer_relevance": sc.answer_relevance,
                "hallucination_rate": sc.hallucination_rate,
            },
        }

    return app


# Module-level app for `uvicorn rag_eval.api:app`.
app = create_app()
