"""Basic vs. enterprise on the same question (FR-3.8, AC-L3-8).

Demonstrates the retrieval-quality gap Lab 2 flagged (R7/R5 in its risk
register): a chunk containing the answer verbatim, phrased so a
deterministic-hash vector embedding has no reason to favour it, is still found
via BM25/hybrid retrieval in enterprise mode.
"""

from __future__ import annotations

import json

from bankassist.config import Settings
from bankassist.llm.stub import StubLLMClient
from bankassist.rag.models import Chunk, DocumentMetadata, VectorRecord
from bankassist.rag.pipeline.basic_pipeline import BasicRagPipeline
from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline
from bankassist.rag.stages.bm25_retriever import BM25Retriever
from bankassist.rag.stages.classifier import QueryClassifier
from bankassist.rag.stages.generator import Generator
from bankassist.rag.stages.query_rewriter import QueryRewriter
from bankassist.rag.stages.vector_retriever import VectorRetriever
from bankassist.rag.stubs import InMemoryVectorStore, StubEmbedder, StubReranker

_DOCUMENT = "03_Raising_Card_Dispute.md"
_TEXT = "the chargeback dispute window is ninety days from the transaction date"
_QUESTION = "chargeback dispute window ninety days"


def _seeded_store(embedder: StubEmbedder) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(
                id=f"{_DOCUMENT}#0",
                values=embedder.embed_query(_TEXT),
                metadata={
                    "document": _DOCUMENT,
                    "title": _DOCUMENT,
                    "category": "Credit Card",
                    "source": "Test",
                    "chunk_index": 0,
                    "text": _TEXT,
                },
            )
        ]
    )
    return store


def test_both_modes_surface_the_same_document_for_an_exact_keyword_query(
    settings: Settings,
) -> None:
    basic_embedder = StubEmbedder(dimensions=16)
    basic_pipeline = BasicRagPipeline(
        settings=settings,
        embedder=basic_embedder,
        store=_seeded_store(basic_embedder),
        llm=StubLLMClient(["the window is 90 days"]),
    )
    basic_result = basic_pipeline.answer(_QUESTION)

    enterprise_embedder = StubEmbedder(dimensions=16)
    llm = StubLLMClient(
        [
            json.dumps({"label": "Procedure", "confidence": 0.9}),
            _QUESTION,
            "the window is 90 days",
        ]
    )
    chunk = Chunk(
        metadata=DocumentMetadata(
            document=_DOCUMENT, title=_DOCUMENT, category="Credit Card", source="Test"
        ),
        text=_TEXT,
        chunk_index=0,
        char_start=0,
        char_end=len(_TEXT),
    )
    enterprise_pipeline = EnterpriseRagPipeline(
        settings=settings,
        classifier=QueryClassifier(llm),
        rewriter=QueryRewriter(llm),
        vector_retriever=VectorRetriever(enterprise_embedder, _seeded_store(enterprise_embedder)),
        bm25_retriever=BM25Retriever([chunk]),
        reranker=StubReranker(),
        generator=Generator(llm),
    )
    enterprise_result = enterprise_pipeline.answer(_QUESTION)

    # BM25 finds the exact-keyword chunk deterministically; the deterministic-hash
    # vector stub may or may not, which is itself the honest point (Lab 2 §11 R7).
    assert _DOCUMENT in [c.metadata.document for c in enterprise_result.bm25_results.results]
    assert enterprise_result.citations == [_DOCUMENT]
    # Both pipelines produce a comparably-shaped grounded answer.
    assert basic_result.grounded is True
    assert enterprise_result.grounded is True
