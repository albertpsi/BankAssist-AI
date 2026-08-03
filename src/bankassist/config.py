"""Application configuration — the only module that reads the environment.

Everything downstream takes a ``Settings`` instance. That single chokepoint is what
lets every test construct an isolated configuration without touching the process
environment, and what later labs extend rather than work around.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bankassist import __version__
from bankassist.errors import ConfigurationError

ModelTier = Literal["fast", "strong"]


class Settings(BaseSettings):
    """Typed application settings, loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `llm_model_fast` would otherwise collide with pydantic's protected
        # `model_` namespace.
        protected_namespaces=(),
    )

    # --- Application ---
    app_name: str = "BankAssist AI"
    app_version: str = __version__
    environment: str = "development"

    # --- LLM provider (ADR-0005) ---
    llm_provider: str = "openai"
    openai_api_key: SecretStr | None = None

    # The economical tier is the default for every operation. The strong tier is
    # optional and reserved for selected LLM-as-judge evaluation cases (NFR-12);
    # when unset, the fast model is used everywhere.
    llm_model_fast: str = "gpt-4o-mini"
    llm_model_strong: str | None = None

    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # --- Observability ---
    log_level: str = "INFO"
    tracing_enabled: bool = True

    @model_validator(mode="after")
    def _require_credential(self) -> Settings:
        """Fail at construction rather than at first LLM call.

        A blank value counts as missing. `.env.example` ships with an empty
        `OPENAI_API_KEY=`, so copying it and forgetting to fill it in yields
        `SecretStr('')` rather than `None` — the single most likely
        misconfiguration, and the one that would otherwise start cleanly and
        fail later with an opaque 401 from the provider.
        """
        if self.llm_provider == "openai" and not self._has_credential():
            raise ConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai', and must "
                "not be blank. Copy .env.example to .env and set it.",
                details={"field": "openai_api_key"},
            )
        return self

    def _has_credential(self) -> bool:
        """True when a non-blank API key is configured."""
        if self.openai_api_key is None:
            return False
        return bool(self.openai_api_key.get_secret_value().strip())

    def model_for_tier(self, tier: ModelTier) -> str:
        """Resolve a tier name to a concrete model id."""
        if tier == "strong":
            return self.llm_model_strong or self.llm_model_fast
        return self.llm_model_fast


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment exactly once."""
    return Settings()
