"""Dense retrieval leg (FR-L3-5.1). Reuses the Lab 2 embedder/store doubles."""

from __future__ import annotations

from bankassist.rag.models import VectorRecord
from bankassist.rag.stages.vector_retriever import VectorRetriever
from bankassist.rag.stubs import InMemoryVectorStore, StubEmbedder


def _seed(store: InMemoryVectorStore, embedder: StubEmbedder, document: str, text: str) -> None:
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
                    "source": "Test Source",
                    "chunk_index": 0,
                    "text": text,
                },
            )
        ]
    )


def test_returns_up_to_top_k_results_ordered_by_score() -> None:
    embedder = StubEmbedder(dimensions=16)
    store = InMemoryVectorStore()
    for i in range(5):
        _seed(store, embedder, f"doc{i}.md", f"clause {i} about chargebacks")

    result = VectorRetriever(embedder, store).execute("chargebacks", top_k=3)

    assert result.query == "chargebacks"
    assert len(result.results) == 3
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_no_results() -> None:
    embedder = StubEmbedder(dimensions=8)
    store = InMemoryVectorStore()

    result = VectorRetriever(embedder, store).execute("anything")

    assert result.results == []
    assert result.latency_ms >= 0.0
