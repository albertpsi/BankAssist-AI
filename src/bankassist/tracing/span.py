"""Span model.

Shaped compatibly with OpenTelemetry (id, parent id, trace id, timing, status,
attributes) so exporting to a real collector later is an exporter change rather
than a rewrite — without taking the OTel dependency now.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SpanType(StrEnum):
    """The kinds of work a trace can contain.

    Only the types the current code actually emits are listed. Later labs add
    their own members (retrieval, agent, tool_call, guardrail, cache) alongside
    the code that emits them.
    """

    REQUEST = "request"
    LLM_CALL = "llm_call"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class Span(BaseModel):
    """A single timed unit of work within a trace."""

    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: str | None = None
    trace_id: str | None = None

    type: SpanType
    name: str

    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None

    status: SpanStatus = SpanStatus.OK
    error_type: str | None = None
    error_message: str | None = None

    attributes: dict[str, Any] = Field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key/value to this span.

        Never attach credentials, unmasked card data, or other customer PII — spans
        are written to disk and screenshotted for the submission document.
        """
        self.attributes[key] = value
