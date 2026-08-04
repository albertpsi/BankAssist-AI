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

ModelTier = Literal["fast", "strong", "classifier"]


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
    # Query classification (Lab 3, FR-L3-3.1). A distinct tier, not folded into
    # "fast", so the lab-mandated model id is configuration on its own line.
    llm_model_classifier: str = "gpt-4.1-mini"

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

    # --- Enterprise retrieval (Lab 3) ---
    retrieval_vector_top_k_enterprise: int = Field(default=20, gt=0)
    retrieval_bm25_top_k: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    # 20, not 10: RRF penalizes a chunk that scores well in only one retrieval leg
    # (e.g. a strong vector match with no BM25 keyword overlap) relative to a
    # weaker chunk both legs agree on. With two 20-item legs the fused list can
    # run 30+ deep, and a single-leg-strong, genuinely-correct chunk can land
    # right at (or just past) a top-10 cut — observed in practice with the KYC
    # OVD-document-list chunk, which BM25 never surfaced (roman-numeral list
    # items, the acronym "OVD") but vector search ranked at position ~6. Cutting
    # at 20 gives that margin without meaningfully increasing reranker cost —
    # the cross-encoder rerank stage is what actually picks the final top_n.
    rerank_candidate_count: int = Field(default=20, gt=0)
    # 8, not 5: with rank-fused selection (reranker.py) a chunk two retrieval
    # legs disagree on can still need a handful of ranks of margin over a
    # cross-encoder-only ordering. Verified empirically against the KYC
    # OVD-document-list case, which needed this before it survived selection.
    rerank_top_n: int = Field(default=8, gt=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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

    # --- Synthetic banking data (Lab 4) ---
    banking_db_path: Path = Path("./data/banking.db")

    # --- Local auth + RBAC (Lab 4, ADR-0010) ---
    # Demo signing secret. Never a production secret — this is a teaching artifact.
    jwt_secret: SecretStr = SecretStr("dev-only-insecure-secret-change-me")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = Field(default=30, gt=0)

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
        if tier == "classifier":
            return self.llm_model_classifier
        return self.llm_model_fast


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment exactly once."""
    return Settings()
