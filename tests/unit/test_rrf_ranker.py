"""Manual Reciprocal Rank Fusion (FR-L3-7)."""

from __future__ import annotations

from bankassist.rag.models import DocumentMetadata
from bankassist.rag.models.retrieval_result import (
    HybridRetrievalResult,
    RetrievalResult,
    ScoredChunk,
)
from bankassist.rag.stages.rrf_ranker import RRFRanker


def _chunk(document: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        text="text",
        metadata=DocumentMetadata(document=document, title=document, category="c", source="s"),
        chunk_index=0,
        score=score,
    )


def test_fusion_score_matches_the_hand_computed_rrf_formula() -> None:
    """A: vector rank 1 only. B: vector rank 2 + bm25 rank 1. C: bm25 rank 2 only.
    With k=60: A=1/61, B=1/62+1/61, C=1/62 -> order B, A, C."""
    hybrid = HybridRetrievalResult(
        vector=RetrievalResult(query="q", results=[_chunk("a.md"), _chunk("b.md")]),
        bm25=RetrievalResult(query="q", results=[_chunk("b.md"), _chunk("c.md")]),
    )

    result = RRFRanker(k=60).execute(hybrid)

    documents = [e.metadata.document for e in result.entries]
    assert documents == ["b.md", "a.md", "c.md"]

    by_doc = {e.metadata.document: e for e in result.entries}
    assert by_doc["a.md"].rrf_score == 1 / 61
    assert by_doc["c.md"].rrf_score == 1 / 62
    assert by_doc["b.md"].rrf_score == 1 / 62 + 1 / 61


def test_vector_only_chunk_carries_no_bm25_rank_or_score() -> None:
    hybrid = HybridRetrievalResult(
        vector=RetrievalResult(query="q", results=[_chunk("a.md")]),
        bm25=RetrievalResult(query="q", results=[]),
    )

    (entry,) = RRFRanker(k=60).execute(hybrid).entries

    assert entry.vector_rank == 1
    assert entry.bm25_rank is None
    assert entry.bm25_score is None


def test_bm25_only_chunk_carries_no_vector_rank_or_score() -> None:
    hybrid = HybridRetrievalResult(
        vector=RetrievalResult(query="q", results=[]),
        bm25=RetrievalResult(query="q", results=[_chunk("a.md")]),
    )

    (entry,) = RRFRanker(k=60).execute(hybrid).entries

    assert entry.bm25_rank == 1
    assert entry.vector_rank is None
    assert entry.vector_score is None


def test_k_is_configurable_and_changes_the_score() -> None:
    hybrid = HybridRetrievalResult(
        vector=RetrievalResult(query="q", results=[_chunk("a.md")]),
        bm25=RetrievalResult(query="q", results=[]),
    )

    low_k = RRFRanker(k=1).execute(hybrid).entries[0].rrf_score
    high_k = RRFRanker(k=1000).execute(hybrid).entries[0].rrf_score

    assert low_k > high_k


def test_output_is_sorted_descending() -> None:
    hybrid = HybridRetrievalResult(
        vector=RetrievalResult(query="q", results=[_chunk("low.md"), _chunk("high.md")]),
        bm25=RetrievalResult(query="q", results=[_chunk("high.md")]),
    )

    scores = [e.rrf_score for e in RRFRanker(k=60).execute(hybrid).entries]

    assert scores == sorted(scores, reverse=True)
