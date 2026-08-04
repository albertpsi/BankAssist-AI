"""SecurityContext — the trustworthy identity every tool call is authorized against.

Built exactly once per request, from a validated JWT only (ADR-0010, FR-24). Never
constructed from a request body field, a tool argument, or anything the LLM generates.
Deliberately excludes secrets: no password hash, no raw token, no signing key.
"""

from __future__ import annotations

from pydantic import BaseModel


class SecurityContext(BaseModel):
    """The authenticated identity and scope for one request."""

    user_id: str
    role: str
    customer_id: str | None
    session_id: str
    request_id: str

    def is_customer(self) -> bool:
        return self.role == "CUSTOMER"
