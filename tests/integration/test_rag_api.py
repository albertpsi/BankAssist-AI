"""`POST /rag/query` through a real ASGI client (FR-L2-9).

The pipeline is injected via `app.state.rag_pipeline` so the route is exercised
without needing Pinecone or OpenAI (NFR-L2-2).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bankassist.api.app import TRACE_HEADER, create_app
from bankassist.config import Settings
from bankassist.errors import VectorStoreError
from bankassist.llm.stub import StubLLMClient
from bankassist.rag.models import VectorRecord
from bankassist.rag.pipeline import BasicRagPipeline
from bankassist.rag.prompts import REFUSAL
from bankassist.rag.stubs import InMemoryVectorStore, StubEmbedder


def _wired_app(
    settings: Settings, llm_responses: list[str] | None = None, seed: bool = True
) -> FastAPI:
    app = create_app(settings)
    embedder = StubEmbedder(dimensions=16)
    store = InMemoryVectorStore()

    if seed:
        vector = embedder.embed_query("the chargeback window is 90 days")
        store.upsert(
            [
                VectorRecord(
                    id="chargeback.md#0",
                    values=vector,
                    metadata={
                        "document": "Chargeback Policy.md",
                        "title": "Chargeback Policy",
                        "category": "Credit Card",
                        "source": "Test Source",
                        "chunk_index": 0,
                        "text": "the chargeback window is 90 days",
                    },
                )
            ]
        )

    app.state.rag_pipeline = BasicRagPipeline(
        settings=settings,
        embedder=embedder,
        store=store,
        llm=StubLLMClient(llm_responses or ["disputes must be raised within 90 days"]),
    )
    return app


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return _wired_app(settings)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_grounded_answer_returns_the_documented_shape(client: TestClient) -> None:
    """AC-L2-8."""
    response = client.post("/rag/query", json={"question": "what is the chargeback window?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "disputes must be raised within 90 days",
        "sources": ["Chargeback Policy.md"],
    }


def test_out_of_corpus_question_returns_the_exact_refusal(settings: Settings) -> None:
    """AC-L2-9."""
    app = _wired_app(settings, seed=False)
    response = TestClient(app).post("/rag/query", json={"question": "anything at all"})

    assert response.status_code == 200
    assert response.json() == {"answer": REFUSAL, "sources": []}


def test_blank_question_is_rejected(client: TestClient) -> None:
    """FR-L2-9.2."""
    response = client.post("/rag/query", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_missing_question_field_is_rejected(client: TestClient) -> None:
    response = client.post("/rag/query", json={})

    assert response.status_code == 422


def test_oversized_question_is_rejected(client: TestClient) -> None:
    """FR-L2-9.2."""
    response = client.post("/rag/query", json={"question": "x" * 2001})

    assert response.status_code == 422


def test_question_at_the_max_length_is_accepted(settings: Settings) -> None:
    app = _wired_app(settings, seed=False)

    response = TestClient(app).post("/rag/query", json={"question": "x" * 2000})

    assert response.status_code == 200


def test_response_carries_a_trace_id(client: TestClient) -> None:
    """FR-L2-9.3."""
    response = client.post("/rag/query", json={"question": "what is the chargeback window?"})

    assert response.headers[TRACE_HEADER]


def test_pipeline_failure_uses_the_error_envelope_not_a_stack_trace(settings: Settings) -> None:
    """FR-L2-9.3."""

    class BoomStore(InMemoryVectorStore):
        def query(self, vector: list[float], top_k: int) -> list:  # type: ignore[override]
            raise VectorStoreError("pinecone query failed: boom", details={})

    app = create_app(settings)
    app.state.rag_pipeline = BasicRagPipeline(
        settings=settings,
        embedder=StubEmbedder(dimensions=8),
        store=BoomStore(),
        llm=StubLLMClient(["unused"]),
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/rag/query", json={"question": "anything"}
    )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "vector_store_error"
    assert body["trace_id"]


def test_health_still_works_with_no_pinecone_configured(settings: Settings) -> None:
    """FR-L2-9.4: only a call to /rag/query needs the Pinecone credential."""
    app = create_app(settings)

    response = TestClient(app).get("/health")

    assert response.status_code == 200


def test_pipeline_is_built_lazily_and_reused(settings: Settings) -> None:
    """FR-L2-9.4: the pipeline is not constructed until the first request."""
    app = create_app(settings)
    assert getattr(app.state, "rag_pipeline", None) is None

    app.state.rag_pipeline = BasicRagPipeline(
        settings=settings,
        embedder=StubEmbedder(dimensions=8),
        store=InMemoryVectorStore(),
        llm=StubLLMClient(["x"]),
    )
    first = app.state.rag_pipeline

    TestClient(app).post("/rag/query", json={"question": "q"})

    assert app.state.rag_pipeline is first


def test_rag_route_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/rag/query" in schema["paths"]
