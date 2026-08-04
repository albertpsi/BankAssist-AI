"""Reranking (FR-L3-8). Uses ``StubReranker`` — no test loads the real
cross-encoder model (NFR-L3-2)."""

from __future__ import annotations

from bankassist.rag.models import DocumentMetadata
from bankassist.rag.models.retrieval_result import RRFEntry, RRFResult
from bankassist.rag.stubs import StubReranker


def _rrf(*documents: str) -> RRFResult:
    return RRFResult(
        entries=[
            RRFEntry(
                text=f"text-{doc}",
                metadata=DocumentMetadata(document=doc, title=doc, category="c", source="s"),
                chunk_index=0,
                rrf_score=1.0 / (index + 1),
            )
            for index, doc in enumerate(documents)
        ]
    )


def test_truncates_ten_candidates_to_five() -> None:
    fused = _rrf(*[f"doc{i}.md" for i in range(10)])

    result = StubReranker().execute("q", fused, top_n=5)

    assert len(result.entries) == 5


def test_each_entry_carries_pre_and_post_rank() -> None:
    fused = _rrf("a.md", "b.md", "c.md")

    result = StubReranker(scores={("text-c.md", 0): 100.0}).execute("q", fused, top_n=3)

    top = result.entries[0]
    assert top.metadata.document == "c.md"
    assert top.pre_rank == 3
    assert top.post_rank == 1


def test_empty_input_returns_empty_output() -> None:
    result = StubReranker().execute("q", RRFResult(entries=[]), top_n=5)

    assert result.entries == []


def test_stub_reranker_records_the_query() -> None:
    reranker = StubReranker()
    fused = _rrf("a.md")

    reranker.execute("the query", fused, top_n=5)

    assert reranker.queries == ["the query"]
