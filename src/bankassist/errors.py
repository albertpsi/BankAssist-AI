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
