"""Local JWT authentication + centralized RBAC (Lab 4, ADR-0010).

Nothing in this package is agent- or LangGraph-aware. It is the trustworthy identity
boundary every tool call is authorized against — built once per request from a
validated JWT, never from a request body field or an LLM-generated tool argument.
"""

from __future__ import annotations

from bankassist.security.authorize import Permission, Role, authorize, permissions_for_role
from bankassist.security.context import SecurityContext
from bankassist.security.jwt_tokens import decode_token, issue_token
from bankassist.security.passwords import hash_password, verify_password

__all__ = [
    "Permission",
    "Role",
    "SecurityContext",
    "authorize",
    "decode_token",
    "hash_password",
    "issue_token",
    "permissions_for_role",
    "verify_password",
]
