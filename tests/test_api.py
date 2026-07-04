from fastapi.testclient import TestClient

from rag_eval.api import create_app
from rag_eval.config import RunConfig
from rag_eval.llm import MockLLM
from rag_eval.schemas import Chunk, RetrievalResult, RetrievedChunk


class _FakeRetriever:
    def retrieve(self, query: str, top_k=None) -> RetrievalResult:
        chunk = Chunk(
            chunk_id="doc1::0",
            doc_id="doc1",
            source_path="doc1.md",
            text="Paris is the capital of France.",
            chunk_index=0,
            start_char=0,
            end_char=31,
        )
        return RetrievalResult(
            query=query, retrieved=[RetrievedChunk(chunk=chunk, score=0.9, rank=1)]
        )


def _client() -> TestClient:
    app = create_app(RunConfig())
    # Inject fakes so no embedding model / LLM / vector store is built.
    app.state.service._retriever = _FakeRetriever()
    app.state.service._llm = MockLLM("The capital is Paris [1].", name="mock:test")
    return TestClient(app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_config_returns_summary():
    r = _client().get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "embedding" in body and "llm" in body


def test_ask_returns_answer_with_resolved_citation():
    r = _client().post("/ask", json={"question": "What is the capital of France?"})
    assert r.status_code == 200
    body = r.json()
    assert "Paris" in body["answer"]
    assert body["retrieved"] == 1
    assert body["citations"] == [
        {"marker": 1, "doc_id": "doc1", "chunk_id": "doc1::0", "score": 0.9}
    ]
    assert body["model"] == "mock:test"


def test_ask_rejects_empty_question():
    r = _client().post("/ask", json={"question": ""})
    assert r.status_code == 422  # pydantic validation
