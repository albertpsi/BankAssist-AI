"""Hybrid composition (FR-L3-5.3): pairs the two result sets, alters neither."""

from __future__ import annotations

from bankassist.rag.models import DocumentMetadata
from bankassist.rag.models.retrieval_result import RetrievalResult, ScoredChunk
from bankassist.rag.stages.hybrid_retriever import HybridRetriever


def _chunk(document: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        text="text",
        metadata=DocumentMetadata(document=document, title=document, category="c", source="s"),
        chunk_index=0,
        score=score,
    )


def test_both_result_sets_pass_through_unaltered() -> None:
    vector = RetrievalResult(query="q", results=[_chunk("a.md", 0.9)])
    bm25 = RetrievalResult(query="q", results=[_chunk("b.md", 5.0)])

    hybrid = HybridRetriever().execute(vector, bm25)

    assert hybrid.vector is vector
    assert hybrid.bm25 is bm25


def test_a_chunk_present_in_only_one_list_is_still_represented() -> None:
    vector = RetrievalResult(query="q", results=[])
    bm25 = RetrievalResult(query="q", results=[_chunk("only-bm25.md", 1.0)])

    hybrid = HybridRetriever().execute(vector, bm25)

    assert hybrid.vector.results == []
    assert hybrid.bm25.results[0].metadata.document == "only-bm25.md"
