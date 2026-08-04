"""The enterprise pipeline end-to-end, built entirely from stage-level stubs
(never a real network call) — FR-L3-11, NFR-L3-2."""

from __future__ import annotations

import json

from bankassist.config import Settings
from bankassist.llm.stub import StubLLMClient
from bankassist.rag.models import Chunk, DocumentMetadata, VectorRecord
from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline
from bankassist.rag.stages.bm25_retriever import BM25Retriever
from bankassist.rag.stages.classifier import QueryClassifier
from bankassist.rag.stages.generator import Generator
from bankassist.rag.stages.query_rewriter import QueryRewriter
from bankassist.rag.stages.vector_retriever import VectorRetriever
from bankassist.rag.stubs import InMemoryVectorStore, StubEmbedder, StubReranker


def _chunk(document: str, text: str) -> Chunk:
    return Chunk(
        metadata=DocumentMetadata(
            document=document, title=document, category="Credit Card", source="Test"
        ),
        text=text,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
    )


def _seed_vector_store(
    store: InMemoryVectorStore, embedder: StubEmbedder, document: str, text: str
) -> None:
    vector = embedder.embed_query(text)
    store.upsert(
        [
            VectorRecord(
                id=f"{document}#0",
                values=vector,
                metadata={
                    "document": document,
                    "title": document,
                    "category": "Credit Card",
                    "source": "Test",
                    "chunk_index": 0,
                    "text": text,
                },
            )
        ]
    )


def _build_pipeline(settings: Settings, llm_responses: list[str]) -> EnterpriseRagPipeline:
    llm = StubLLMClient(llm_responses)
    embedder = StubEmbedder(dimensions=16)
    store = InMemoryVectorStore()
    text = "the chargeback dispute window is 90 days"
    _seed_vector_store(store, embedder, "chargeback.md", text)
    chunks = [_chunk("chargeback.md", text)]

    return EnterpriseRagPipeline(
        settings=settings,
        classifier=QueryClassifier(llm),
        rewriter=QueryRewriter(llm),
        vector_retriever=VectorRetriever(embedder, store),
        bm25_retriever=BM25Retriever(chunks),
        reranker=StubReranker(),
        generator=Generator(llm),
    )


def test_pipeline_result_has_every_explainability_field_populated(settings: Settings) -> None:
    pipeline = _build_pipeline(
        settings,
        llm_responses=[
            json.dumps({"label": "Procedure", "confidence": 0.9}),
            "What is the chargeback dispute time limit?",
            "disputes must be raised within 90 days",
        ],
    )

    result = pipeline.answer("How long do I have to dispute a transaction?")

    assert result.original_question == "How long do I have to dispute a transaction?"
    assert result.classification.label == "Procedure"
    assert result.rewritten_question == "What is the chargeback dispute time limit?"
    assert result.vector_results.results
    assert result.bm25_results.results
    assert result.rrf_results.entries
    assert result.reranked_results.entries
    assert result.prompt_context.chunk_count > 0
    assert result.generated_answer == "disputes must be raised within 90 days"
    assert result.grounded is True
    assert result.citations == ["chargeback.md"]
    assert set(result.latencies) >= {
        "classification",
        "rewrite",
        "vector_retrieval",
        "bm25_retrieval",
        "rrf",
        "rerank",
        "generation",
    }


def test_grounded_end_to_end_answer_with_citations(settings: Settings) -> None:
    pipeline = _build_pipeline(
        settings,
        llm_responses=[
            json.dumps({"label": "Policy", "confidence": 0.8}),
            "chargeback dispute window",
            "the window is 90 days",
        ],
    )

    result = pipeline.answer("chargeback window")

    assert result.grounded is True
    assert result.citations == ["chargeback.md"]
