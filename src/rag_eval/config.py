"""Config-driven pipeline.

A single ``RunConfig`` threads through ingestion, retrieval, answering, and
evaluation. Because every stage is a pure function of this config, experiments
(different chunk size, k, embedding model, LLM) are reproducible and easy to
sweep. Config is loaded from YAML; secrets never live here — API keys come from
the environment (see ``.env.example``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

EmbeddingProvider = Literal["bge", "openai", "voyage"]
LLMProvider = Literal["groq", "openai", "anthropic"]


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = "bge"
    model: str = "BAAI/bge-small-en-v1.5"
    # Larger batches speed up local embedding; tune per hardware.
    batch_size: int = 32


class LLMConfig(BaseModel):
    provider: LLMProvider = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(512, description="Target chunk size in tokens (approx).")
    chunk_overlap: int = Field(64, description="Token overlap between adjacent chunks.")


class RetrievalConfig(BaseModel):
    top_k: int = 5


class StorageConfig(BaseModel):
    persist_dir: str = ".chroma"
    collection: str = "rag_eval"


class PathsConfig(BaseModel):
    corpus_dir: str = "data/corpus"
    eval_set: str = "data/eval/eval_set.jsonl"
    runs_dir: str = "runs"


class RunConfig(BaseModel):
    """The single source of truth for a pipeline run."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # Judge defaults to the same provider/model as the answerer but can be
    # pointed at a different (often stronger) model to reduce self-preference bias.
    judge: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def with_overrides(self, **dotted: Any) -> RunConfig:
        """Return a copy with dotted-key overrides applied.

        Example: ``cfg.with_overrides(**{"llm.provider": "anthropic", "retrieval.top_k": 8})``.
        Enables CLI flags and config sweeps without mutating the original.
        """
        data = self.model_dump()
        for dotted_key, value in dotted.items():
            if value is None:
                continue
            keys = dotted_key.split(".")
            cursor = data
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = value
        return RunConfig.model_validate(data)

    def summary(self) -> dict[str, str]:
        """Compact, human-readable identity of this config for scorecards/tables."""
        return {
            "embedding": f"{self.embedding.provider}:{self.embedding.model}",
            "llm": f"{self.llm.provider}:{self.llm.model}",
            "judge": f"{self.judge.provider}:{self.judge.model}",
            "chunk_size": str(self.chunking.chunk_size),
            "chunk_overlap": str(self.chunking.chunk_overlap),
            "top_k": str(self.retrieval.top_k),
        }
