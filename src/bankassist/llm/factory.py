"""Provider selection.

The single place that knows which providers exist. Adding an Anthropic adapter
later (ADR-0005) means one branch here and one new module — no call site changes.
"""

from __future__ import annotations

from bankassist.config import Settings
from bankassist.errors import ConfigurationError
from bankassist.llm.base import LLMClient
from bankassist.llm.openai_client import OpenAIClient
from bankassist.tracing.tracer import Tracer

SUPPORTED_PROVIDERS = ("openai",)


def build_llm_client(settings: Settings, tracer: Tracer | None = None) -> LLMClient:
    """Return the client implied by configuration."""
    if settings.llm_provider == "openai":
        return OpenAIClient(settings, tracer)

    raise ConfigurationError(
        f"Unsupported LLM_PROVIDER {settings.llm_provider!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}.",
        details={"field": "llm_provider", "value": settings.llm_provider},
    )
