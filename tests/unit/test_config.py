"""Configuration behaviour (FR-L1-2)."""

from __future__ import annotations

import pytest

from bankassist.config import Settings, get_settings
from bankassist.errors import ConfigurationError


def test_import_requires_no_environment() -> None:
    """AC-L1-1: importing the package must not need configuration."""
    import bankassist

    assert bankassist.__version__


def test_defaults_applied(settings: Settings) -> None:
    assert settings.llm_provider == "openai"
    assert settings.llm_max_tokens == 4096
    assert settings.tracing_enabled is True


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("LLM_MODEL_FAST", "model-from-env")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    loaded = Settings(_env_file=None)

    assert loaded.llm_model_fast == "model-from-env"
    assert loaded.log_level == "DEBUG"


def test_missing_api_key_raises_configuration_error() -> None:
    """AC-L1-2: fail loudly at construction, naming the offending field."""
    with pytest.raises(ConfigurationError) as excinfo:
        Settings(llm_provider="openai", openai_api_key=None, _env_file=None)

    assert excinfo.value.details["field"] == "openai_api_key"
    assert "OPENAI_API_KEY" in excinfo.value.message


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_api_key_is_treated_as_missing(blank: str) -> None:
    """`.env.example` ships `OPENAI_API_KEY=`, so a copied-but-unfilled .env
    yields SecretStr('') rather than None. That must fail at construction, not
    at the first provider call."""
    with pytest.raises(ConfigurationError) as excinfo:
        Settings(llm_provider="openai", openai_api_key=blank, _env_file=None)

    assert excinfo.value.details["field"] == "openai_api_key"


def test_blank_api_key_from_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The realistic path: the value arrives from the environment, not a literal."""
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ConfigurationError, match="blank"):
        Settings(_env_file=None)


def test_strong_tier_falls_back_to_fast_when_unset(settings: Settings) -> None:
    """AC-L1-3: the strong tier is optional by design."""
    assert settings.llm_model_strong is None
    assert settings.model_for_tier("strong") == settings.llm_model_fast
    assert settings.model_for_tier("fast") == settings.llm_model_fast


def test_strong_tier_used_when_configured() -> None:
    configured = Settings(
        openai_api_key="sk-test", llm_model_strong="a-stronger-model", _env_file=None
    )

    assert configured.model_for_tier("strong") == "a-stronger-model"


def test_api_key_never_appears_in_repr_or_str() -> None:
    """AC-L1-4: SecretStr must keep the credential out of every rendering."""
    secret = "sk-super-secret-value-12345"
    configured = Settings(openai_api_key=secret, _env_file=None)

    assert secret not in repr(configured)
    assert secret not in str(configured)
    assert secret not in repr(configured.openai_api_key)
    # It is still retrievable deliberately, for the one caller that needs it.
    assert configured.openai_api_key is not None
    assert configured.openai_api_key.get_secret_value() == secret


def test_api_key_not_leaked_by_model_dump() -> None:
    """A serialized Settings must not carry the raw credential either."""
    secret = "sk-super-secret-value-12345"
    configured = Settings(openai_api_key=secret, _env_file=None)

    assert secret not in str(configured.model_dump())


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-L1-2.7: the environment is read once per process."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()
