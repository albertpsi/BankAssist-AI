"""API request and response models.

One error envelope for every failure path, so clients parse a single shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
