"""Cross-encoder reranking output (FR-L3-8)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bankassist.rag.models import DocumentMetadata


class RerankEntry(BaseModel):
    """One surviving chunk, with its rank/score before and after reranking."""

    text: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    pre_rank: int = Field(ge=1)
    pre_score: float
    post_rank: int = Field(ge=1)
    post_score: float


class RerankResult(BaseModel):
    """The top ``top_n`` chunks after reranking, best first (FR-L3-8.2)."""

    entries: list[RerankEntry] = Field(default_factory=list)
