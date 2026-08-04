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


# --- Lab 4: local auth (ADR-0010) ---


class LoginRequest(BaseModel):
    """``POST /api/v1/auth/login`` request body (FR-22)."""

    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    """``POST /api/v1/auth/login`` response body. Never echoes the password."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    role: str
    customer_id: str | None = None


# --- Lab 4: multi-agent chat (FR-19) ---

MAX_MESSAGE_CHARS = 2000


class ExecutionEventSchema(BaseModel):
    """API projection of ``bankassist.execution_event.ExecutionEvent``."""

    event_type: str
    node_id: str
    node_type: str
    label: str
    status: str
    timestamp: str
    summary: str


class AgentChatRequest(BaseModel):
    """``POST /api/v1/agent/chat`` request body (FR-19)."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    customer_id: str | None = None
    session_id: str = Field(min_length=1, max_length=100)

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class TransactionOption(BaseModel):
    """One clickable choice when the Dispute Agent is asking which transaction.

    Deliberately narrow: only what the UI needs to render a button and echo an
    unambiguous follow-up message — not the full ``Transaction`` tool model.
    """

    transaction_id: str
    merchant: str
    amount_rupees: float
    txn_date: str


class AgentChatResponse(BaseModel):
    """``POST /api/v1/agent/chat`` response body (FR-19).

    Never contains chain-of-thought, hidden prompts, or raw LangGraph state.
    """

    answer: str
    agent: str
    session_id: str
    status: Literal["completed", "waiting_approval", "failed"]
    approval_required: bool = False
    sources: list[str] = Field(default_factory=list)
    execution_events: list[ExecutionEventSchema] = Field(default_factory=list)
    available_transactions: list[TransactionOption] | None = None


class AgentResumeRequest(BaseModel):
    """Resumes an interrupted thread with a human approval decision (FR-20)."""

    session_id: str = Field(min_length=1, max_length=100)
    approved: bool
