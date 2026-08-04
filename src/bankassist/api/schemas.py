"""API request and response models.

One error envelope for every failure path, so clients parse a single shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 2000


class HealthResponse(BaseModel):
    """Liveness payload. Contains no credential and no customer data."""

    status: Literal["ok"] = "ok"
    app: str
    version: str
    environment: str
    llm_provider: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """The single error envelope used by every handler."""

    error: ErrorDetail
    trace_id: str | None = None


RagMode = Literal["basic", "enterprise"]


class RagQueryRequest(BaseModel):
    """``POST /rag/query`` request body (FR-L2-9.1, extended by FR-L3-2)."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: RagMode = "basic"

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """FR-L2-9.2: whitespace-only is blank, not a 1-character question."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class RagQueryResponse(BaseModel):
    """``POST /rag/query`` response body (FR-L2-9.1). Unchanged by Lab 3 —

    ``mode: "basic"`` (or omitted) returns exactly this shape, byte-identical
    to Lab 2 (NFR-L3-4).
    """

    answer: str
    sources: list[str]


class EnterpriseClassificationSummary(BaseModel):
    """The explainability fields the enterprise response surfaces (FR-L3-12.1).

    Deliberately narrower than ``ClassificationResult`` / ``PipelineResult`` —
    the API response is a summary for the UI, not the full explainability
    object those internal types carry.
    """

    label: str
    confidence: float


class EnterpriseRagQueryResponse(BaseModel):
    """``POST /rag/query`` response body when ``mode: "enterprise"`` (FR-L3-12.1)."""

    answer: str
    sources: list[str]
    mode: Literal["enterprise"] = "enterprise"
    classification: EnterpriseClassificationSummary
    rewritten_question: str
