"""Metadata filtering and prompt-construction request/result objects (FR-L3-6, FR-L3-9)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bankassist.llm.base import LLMMessage
from bankassist.rag.models.rerank_result import RerankResult


class MetadataFilters(BaseModel):
    """Optional exact-match narrowing, ANDed across whichever fields are set.

    ``None`` on every field (the default) is a no-op passthrough (FR-L3-6.2).
    """

    category: str | None = None
    document: str | None = None
    source: str | None = None

    def is_empty(self) -> bool:
        return self.category is None and self.document is None and self.source is None


class PromptBuildRequest(BaseModel):
    """What ``PromptBuilder`` needs to assemble the enterprise prompt."""

    original_question: str
    rewritten_question: str
    reranked: RerankResult


class PromptContext(BaseModel):
    """The assembled prompt, plus what the lab brief wants logged about it."""

    messages: list[LLMMessage]
    chunk_count: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
