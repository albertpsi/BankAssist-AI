"""`mode` dispatch on `POST /rag/query` and its `/api/v1` alias (FR-L3-2, FR-L3-12).

A separate file from `test_rag_api.py` so Lab 2's suite stays untouched — its
assertions are the proof that `basic`/omitted-mode behaviour is byte-identical
to before this lab (NFR-L3-4).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bankassist.api.app import create_app
from bankassist.config import Settings
from bankassist.llm.stub import StubLLMClient
from bankassist.rag.models import Chunk, DocumentMetadata, VectorRecord
from bankassist.rag.pipeline import BasicRagPipeline
from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline
from bankassist.rag.stages.bm25_retriever import BM25Retriever
from bankassist.rag.stages.classifier import QueryClassifier
from bankassist.rag.stages.generator import Generator
from bankassist.rag.stages.query_rewriter import QueryRewriter
from bankassist.rag.stages.vector_retriever import VectorRetriever
from bankassist.rag.stubs import InMemoryVectorStore, StubEmbedder, StubReranker


def _wired_app(settings: Settings, llm_responses: list[str]) -> FastAPI:
    app = create_app(settings)

    basic_llm = StubLLMClient(["a basic-mode answer"])
    embedder = StubEmbedder(dimensions=16)
    store = InMemoryVectorStore()
    text = "the chargeback dispute window is 90 days"
    vector = embedder.embed_query(text)
    store.upsert(
        [
            VectorRecord(
                id="chargeback.md#0",
                values=vector,
                metadata={
                    "document": "chargeback.md",
                    "title": "chargeback.md",
                    "category": "Credit Card",
                    "source": "Test",
                    "chunk_index": 0,
                    "text": text,
                },
            )
        ]
    )
    app.state.rag_pipeline = BasicRagPipeline(
        settings=settings, embedder=embedder, store=store, llm=basic_llm
    )

    enterprise_llm = StubLLMClient(llm_responses)
    enterprise_embedder = StubEmbedder(dimensions=16)
    enterprise_store = InMemoryVectorStore()
    enterprise_embedder.embed_query(text)
    enterprise_store.upsert(
        [
            VectorRecord(
                id="chargeback.md#0",
                values=enterprise_embedder.embed_query(text),
                metadata={
                    "document": "chargeback.md",
                    "title": "chargeback.md",
                    "category": "Credit Card",
                    "source": "Test",
                    "chunk_index": 0,
                    "text": text,
                },
            )
        ]
    )
    chunk = Chunk(
        metadata=DocumentMetadata(
            document="chargeback.md", title="chargeback.md", category="Credit Card", source="Test"
        ),
        text=text,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
    )
    app.state.enterprise_rag_pipeline = EnterpriseRagPipeline(
        settings=settings,
        classifier=QueryClassifier(enterprise_llm),
        rewriter=QueryRewriter(enterprise_llm),
        vector_retriever=VectorRetriever(enterprise_embedder, enterprise_store),
        bm25_retriever=BM25Retriever([chunk]),
        reranker=StubReranker(),
        generator=Generator(enterprise_llm),
    )
    return app


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return _wired_app(
        settings,
        llm_responses=[
            json.dumps({"label": "Procedure", "confidence": 0.9}),
            "chargeback dispute window",
            "disputes must be raised within 90 days",
        ],
    )


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_mode_defaults_to_basic(client: TestClient) -> None:
    response = client.post("/rag/query", json={"question": "what is the chargeback window?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "a basic-mode answer", "sources": ["chargeback.md"]}


def test_mode_basic_is_explicit_and_identical_to_the_default(client: TestClient) -> None:
    response = client.post(
        "/rag/query", json={"question": "what is the chargeback window?", "mode": "basic"}
    )

    assert response.json() == {"answer": "a basic-mode answer", "sources": ["chargeback.md"]}


def test_mode_enterprise_returns_the_extended_response_shape(client: TestClient) -> None:
    response = client.post(
        "/rag/query",
        json={"question": "how long do I have to dispute a transaction?", "mode": "enterprise"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "disputes must be raised within 90 days"
    assert body["sources"] == ["chargeback.md"]
    assert body["mode"] == "enterprise"
    assert body["classification"] == {"label": "Procedure", "confidence": 0.9}
    assert body["rewritten_question"] == "chargeback dispute window"


def test_invalid_mode_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/rag/query", json={"question": "anything", "mode": "not-a-real-mode"}
    )

    assert response.status_code == 422


def test_api_v1_alias_also_serves_the_route(client: TestClient) -> None:
    response = client.post(
        "/api/v1/rag/query", json={"question": "what is the chargeback window?"}
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "a basic-mode answer", "sources": ["chargeback.md"]}
