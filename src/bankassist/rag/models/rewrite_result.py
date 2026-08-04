"""Query rewrite output (FR-L3-4)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewriteResult(BaseModel):
    """The original question is never discarded — only the rewrite is new."""

    original_question: str
    rewritten_question: str
    latency_ms: float = Field(default=0.0, ge=0.0)
    fallback_used: bool = False
