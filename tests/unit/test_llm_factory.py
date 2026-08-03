"""Provider selection (FR-L1-6.6)."""

from __future__ import annotations

import pytest

from bankassist.config import Settings
from bankassist.errors import ConfigurationError
from bankassist.llm.factory import build_llm_client
from bankassist.llm.openai_client import OpenAIClient


def test_openai_provider_builds_the_openai_client(settings: Settings) -> None:
    assert isinstance(build_llm_client(settings), OpenAIClient)


def test_unknown_provider_raises_configuration_error(settings: Settings) -> None:
    """AC: the factory owns the provider set and rejects anything outside it."""
    unknown = settings.model_copy(update={"llm_provider": "not-a-provider"})

    with pytest.raises(ConfigurationError) as excinfo:
        build_llm_client(unknown)

    assert excinfo.value.details["value"] == "not-a-provider"
    assert "openai" in excinfo.value.message


def test_client_without_a_key_fails_at_construction(settings: Settings) -> None:
    """Reachable when a caller bypasses the factory with other-provider settings."""
    keyless = settings.model_copy(update={"openai_api_key": None})

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIClient(keyless)
