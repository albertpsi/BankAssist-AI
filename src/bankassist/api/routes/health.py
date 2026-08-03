"""Liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankassist.api.schemas import HealthResponse
from bankassist.config import Settings

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health(request: Request) -> HealthResponse:
    """Report that the application is up and how it is configured.

    Deliberately does not call the LLM provider: this must stay usable as a
    liveness probe when the provider is down or the key is invalid. It reports
    *which* provider is configured, never the credential.
    """
    settings: Settings = request.app.state.settings
    return HealthResponse(
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
    )
