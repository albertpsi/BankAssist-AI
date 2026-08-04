"""Sparse (BM25) retrieval leg (FR-L3-5.2)."""

from __future__ import annotations

from bankassist.rag.models import Chunk, DocumentMetadata
from bankassist.rag.stages.bm25_retriever import BM25Retriever


def _chunk(document: str, text: str, index: int = 0) -> Chunk:
    return Chunk(
        metadata=DocumentMetadata(
            document=document, title=document, category="Credit Card", source="Test"
        ),
        text=text,
        chunk_index=index,
        char_start=0,
        char_end=len(text),
    )


def test_exact_keyword_match_ranks_top_even_where_vector_search_would_miss() -> None:
    """The retrieval-quality claim AC-L3-8 depends on: an exact term the vector
    stub (deterministic hash embeddings) has no reason to favour still wins BM25."""
    chunks = [
        _chunk("chargeback.md", "the chargeback dispute window is ninety days", index=0),
        _chunk("kyc.md", "KYC requires a passport or Aadhaar for onboarding", index=0),
        _chunk("rewards.md", "reward points expire after two years of inactivity", index=0),
    ]

    result = BM25Retriever(chunks).execute("chargeback dispute window", top_k=20)

    assert result.results
    assert result.results[0].metadata.document == "chargeback.md"


def test_empty_corpus_returns_no_results() -> None:
    result = BM25Retriever([]).execute("anything")

    assert result.results == []


def test_query_with_no_overlap_returns_no_results() -> None:
    chunks = [_chunk("a.md", "alpha beta gamma")]

    result = BM25Retriever(chunks).execute("zzzznomatch")

    assert result.results == []


def test_results_are_capped_at_top_k() -> None:
    chunks = [_chunk(f"doc{i}.md", "fee schedule fee schedule fee", index=0) for i in range(5)]

    result = BM25Retriever(chunks).execute("fee schedule", top_k=2)

    assert len(result.results) == 2
