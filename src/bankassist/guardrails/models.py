"""``GuardrailResult`` — the one contract every guardrail layer returns (Lab 5).

Mirrors the existing ``BankAssistError`` convention (a ``code``-like id, a safe
``reason``, structured ``details``) rather than inventing a parallel shape. Nothing
downstream — ``ExecutionEvent``, the API response, the Streamlit UI — ever sees a
NeMo-specific or regex-specific type; everything is normalized to this model first.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GuardrailAction(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REDACT = "REDACT"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class GuardrailCategory(StrEnum):
    INPUT = "INPUT"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    OWNERSHIP = "OWNERSHIP"
    TOOL = "TOOL"
    OUTPUT = "OUTPUT"
    RAG = "RAG"


class GuardrailResult(BaseModel):
    """The verdict from one guardrail check.

    ``reason`` is user-safe and is what an ``ExecutionEvent.summary`` may quote.
    ``internal_reason`` may carry more technical detail (e.g. which pattern matched)
    and is for logs only — never surfaced to the end user or the API response.
    """

    allowed: bool
    guardrail_id: str
    category: GuardrailCategory
    reason: str
    internal_reason: str | None = None
    severity: str = Field(default="low", pattern="^(low|medium|high)$")
    action: GuardrailAction
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def allow(
        cls, guardrail_id: str, category: GuardrailCategory, reason: str = ""
    ) -> GuardrailResult:
        return cls(
            allowed=True,
            guardrail_id=guardrail_id,
            category=category,
            reason=reason or "Passed.",
            action=GuardrailAction.ALLOW,
        )

    @classmethod
    def block(
        cls,
        guardrail_id: str,
        category: GuardrailCategory,
        reason: str,
        *,
        severity: str = "high",
        internal_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        return cls(
            allowed=False,
            guardrail_id=guardrail_id,
            category=category,
            reason=reason,
            internal_reason=internal_reason,
            severity=severity,
            action=GuardrailAction.BLOCK,
            metadata=metadata or {},
        )
