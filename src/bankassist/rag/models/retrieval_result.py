"""Per-retriever and fused-ranking results (FR-L3-5, FR-L3-7)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bankassist.rag.models import DocumentMetadata


class ScoredChunk(BaseModel):
    """One retriever's hit: the chunk, and how well it scored."""

    text: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    score: float


class RetrievalResult(BaseModel):
    """One retriever's full result set for one query."""

    query: str
    results: list[ScoredChunk] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)


class HybridRetrievalResult(BaseModel):
    """The vector and BM25 result sets, paired but not yet fused (FR-L3-5.3)."""

    vector: RetrievalResult
    bm25: RetrievalResult


class RRFEntry(BaseModel):
    """One chunk's fused ranking, with both legs' contribution kept visible."""

    text: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float


class RRFResult(BaseModel):
    """The fused ranking, sorted descending by ``rrf_score`` (FR-L3-7.3)."""

    entries: list[RRFEntry] = Field(default_factory=list)
