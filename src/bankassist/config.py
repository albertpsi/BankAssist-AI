"""Application configuration — the only module that reads the environment.

Everything downstream takes a ``Settings`` instance. That single chokepoint is what
lets every test construct an isolated configuration without touching the process
environment, and what later labs extend rather than work around.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
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

    # --- Corpus (Lab 2) ---
    # Holds markdown/, metadata/, and pdf/ subdirectories.
    policy_corpus_dir: Path = Path("./data/policies")

    # --- Embeddings (Lab 2, ADR-0007) ---
    embedding_model: str = "text-embedding-3-small"
    # Must equal the Pinecone index dimension, which is fixed at index creation.
    embedding_dimensions: int = Field(default=1536, gt=0)
    embedding_batch_size: int = Field(default=100, gt=0)

    # --- Chunking (Lab 2) ---
    # Characters, not tokens. See `_validate_chunking` for the invariants.
    chunk_size_chars: int = Field(default=800, gt=0)
    chunk_min_chars: int = Field(default=700, gt=0)
    chunk_max_chars: int = Field(default=900, gt=0)
    chunk_overlap_chars: int = Field(default=120, ge=0)

    # --- Retrieval (Lab 2) ---
    retrieval_top_k: int = Field(default=5, gt=0)

    # --- Vector store (Lab 2, ADR-0007) ---
    # Deliberately *not* validated here. Unlike the OpenAI key, this credential is
    # needed only by the code that talks to Pinecone — requiring it at startup
    # would take down /health and the entire test suite on a machine that has no
    # Pinecone account. `PineconeVectorStore` fails loudly at construction instead.
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "bankassist-policies"
    pinecone_namespace: str = "bank-policies"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Clients ---
    # Where the Streamlit UI reaches the API. The UI is a separate process.
    api_base_url: str = "http://127.0.0.1:8000"

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

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Reject chunk settings the chunker could not honour.

        Two invariants, both of which are silent corruption rather than a crash if
        they are violated: a target outside its own accepted window means every
        chunk is out of spec, and an overlap at or above the minimum size means the
        next chunk starts at or before the current one — an infinite loop.
        """
        if not self.chunk_min_chars <= self.chunk_size_chars <= self.chunk_max_chars:
            raise ConfigurationError(
                "CHUNK_SIZE_CHARS must lie within [CHUNK_MIN_CHARS, CHUNK_MAX_CHARS]. "
                f"Got min={self.chunk_min_chars}, size={self.chunk_size_chars}, "
                f"max={self.chunk_max_chars}.",
                details={"field": "chunk_size_chars"},
            )
        if self.chunk_overlap_chars >= self.chunk_min_chars:
            raise ConfigurationError(
                "CHUNK_OVERLAP_CHARS must be smaller than CHUNK_MIN_CHARS, or chunking "
                f"cannot advance. Got overlap={self.chunk_overlap_chars}, "
                f"min={self.chunk_min_chars}.",
                details={"field": "chunk_overlap_chars"},
            )
        return self

    def _has_credential(self) -> bool:
        """True when a non-blank API key is configured."""
        if self.openai_api_key is None:
            return False
        return bool(self.openai_api_key.get_secret_value().strip())

    def has_pinecone_credential(self) -> bool:
        """True when a non-blank Pinecone key is configured."""
        if self.pinecone_api_key is None:
            return False
        return bool(self.pinecone_api_key.get_secret_value().strip())

    @property
    def markdown_dir(self) -> Path:
        """Where the ingestion input lives."""
        return self.policy_corpus_dir / "markdown"

    @property
    def metadata_dir(self) -> Path:
        """Where each document's metadata sidecar lives, matched by file stem."""
        return self.policy_corpus_dir / "metadata"

    def model_for_tier(self, tier: ModelTier) -> str:
        """Resolve a tier name to a concrete model id."""
        if tier == "strong":
            return self.llm_model_strong or self.llm_model_fast
        return self.llm_model_fast


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment exactly once."""
    return Settings()
