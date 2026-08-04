"""Optional metadata narrowing (FR-L3-6)."""

from __future__ import annotations

from bankassist.rag.models import DocumentMetadata
from bankassist.rag.models.retrieval_context import MetadataFilters
from bankassist.rag.models.retrieval_result import (
    HybridRetrievalResult,
    RetrievalResult,
    ScoredChunk,
)
from bankassist.rag.stages.metadata_filter import MetadataFilter


def _chunk(document: str, category: str, source: str) -> ScoredChunk:
    return ScoredChunk(
        text="text",
        metadata=DocumentMetadata(
            document=document, title=document, category=category, source=source
        ),
        chunk_index=0,
        score=1.0,
    )


def _hybrid(*chunks: ScoredChunk) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        vector=RetrievalResult(query="q", results=list(chunks)),
        bm25=RetrievalResult(query="q", results=list(chunks)),
    )


def test_none_filter_is_a_no_op() -> None:
    hybrid = _hybrid(_chunk("a.md", "KYC", "s"))

    result = MetadataFilter().execute(hybrid, None)

    assert result is hybrid


def test_empty_filters_object_is_a_no_op() -> None:
    hybrid = _hybrid(_chunk("a.md", "KYC", "s"))

    result = MetadataFilter().execute(hybrid, MetadataFilters())

    assert result is hybrid


def test_category_filter_narrows_to_matching_chunks() -> None:
    hybrid = _hybrid(_chunk("kyc.md", "KYC", "s"), _chunk("card.md", "Credit Card", "s"))

    result = MetadataFilter().execute(hybrid, MetadataFilters(category="KYC"))

    assert [c.metadata.document for c in result.vector.results] == ["kyc.md"]
    assert [c.metadata.document for c in result.bm25.results] == ["kyc.md"]


def test_combined_filters_and_together() -> None:
    hybrid = _hybrid(
        _chunk("a.md", "KYC", "Official"),
        _chunk("b.md", "KYC", "Unofficial"),
    )

    result = MetadataFilter().execute(hybrid, MetadataFilters(category="KYC", source="Official"))

    assert [c.metadata.document for c in result.vector.results] == ["a.md"]


def test_filter_matching_nothing_returns_empty_not_an_error() -> None:
    hybrid = _hybrid(_chunk("a.md", "KYC", "s"))

    result = MetadataFilter().execute(hybrid, MetadataFilters(category="Nonexistent"))

    assert result.vector.results == []
    assert result.bm25.results == []
