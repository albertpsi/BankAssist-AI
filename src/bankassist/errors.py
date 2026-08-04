"""Application exception hierarchy.

Each error carries the HTTP status it maps to, so the API layer never needs a
translation table that can drift from the exceptions it translates.

Only the failure modes the Lab 1 foundation can actually raise are defined here.
Later labs add their own alongside the code that raises them.
"""

from __future__ import annotations

from typing import Any


class BankAssistError(Exception):
    """Base for every error this application raises deliberately.

    ``details`` is surfaced to the caller in the API error envelope, so it must
    never contain credentials, customer data, or internal paths.
    """

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(BankAssistError):
    """Settings are missing, malformed, or internally inconsistent."""

    code = "configuration_error"
    http_status = 500


class LLMError(BankAssistError):
    """An LLM provider call failed.

    Wraps provider SDK exceptions so they do not leak past the ``llm`` package.
    502 rather than 500: the failure is in an upstream dependency, not in us.
    """

    code = "llm_error"
    http_status = 502


class IngestionError(BankAssistError):
    """The policy corpus could not be read.

    A corpus problem is an operator problem — a missing sidecar, an unparseable
    JSON file, a required key absent — so the message always names the file.
    Raised only by the ingestion path, which is a CLI, never a request.
    """

    code = "ingestion_error"
    http_status = 500


class EmbeddingError(BankAssistError):
    """An embeddings provider call failed.

    Separate from ``LLMError`` because the two have different failure modes and
    different cost accounting, even though they share a provider and an SDK.
    """

    code = "embedding_error"
    http_status = 502


class VectorStoreError(BankAssistError):
    """A vector store operation failed.

    Wraps Pinecone SDK exceptions so they do not leak past the ``rag`` package.
    """

    code = "vector_store_error"
    http_status = 502


class AuthenticationError(BankAssistError):
    """Credentials were missing, invalid, or the token was expired/tampered.

    Deliberately generic (ADR-0010, FR-22): never reveals whether the username or
    the password was wrong.
    """

    code = "authentication_error"
    http_status = 401


class AuthorizationError(BankAssistError):
    """The authenticated identity lacks the permission or ownership for this action.

    Raised before any sensitive tool executes (FR-27); ``details`` may name the
    permission that was denied but never the other party's data.
    """

    code = "authorization_error"
    http_status = 403


class NoPendingApprovalError(BankAssistError):
    """A resume was requested for a session with no interrupted graph waiting.

    Covers both "never paused" and "already resumed once" (duplicate-resume
    protection, FR-15/AC-6) — a client-state conflict, not a server fault.
    """

    code = "no_pending_approval"
    http_status = 409
