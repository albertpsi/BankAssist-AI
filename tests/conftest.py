"""Shared fixtures.

Every fixture here builds isolated objects. Nothing in the suite reads the real
process environment, calls a network, or needs an API key (AC-L1-17).
"""

from __future__ import annotations

import pytest

from bankassist.config import Settings
from bankassist.tracing.tracer import InMemoryTracer


@pytest.fixture
def settings() -> Settings:
    """Valid settings built in-process, never from the ambient environment."""
    return Settings(
        openai_api_key="sk-test-not-a-real-key",
        llm_model_fast="test-fast-model",
        llm_model_strong=None,
        environment="test",
        _env_file=None,
    )


@pytest.fixture
def tracer() -> InMemoryTracer:
    return InMemoryTracer()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider variables so an ambient key can never influence a test."""
    for name in ("OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_FAST", "LLM_MODEL_STRONG"):
        monkeypatch.delenv(name, raising=False)
