"""FastAPI dependency that turns a Bearer token into a ``SecurityContext`` (FR-23/24)."""

from __future__ import annotations

import uuid

from fastapi import Header, Request

from bankassist.config import Settings
from bankassist.context import get_trace_id
from bankassist.errors import AuthenticationError
from bankassist.security.context import SecurityContext
from bankassist.security.jwt_tokens import decode_token


def require_security_context(
    request: Request,
    authorization: str | None = Header(default=None),
) -> SecurityContext:
    """Validate the ``Authorization: Bearer <token>`` header and build the context.

    ``session_id`` is read from the request body's field of the same name where the
    route defines one, and defaults to a fresh id here; individual routes needing the
    caller-supplied session id read it from the parsed body, not from this dependency.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("Missing or malformed Authorization header.")

    token = authorization.split(" ", 1)[1].strip()
    settings: Settings = request.app.state.settings
    claims = decode_token(token, settings=settings)

    return SecurityContext(
        user_id=str(claims.get("sub", "")),
        role=str(claims.get("role", "")),
        customer_id=claims.get("customer_id"),
        session_id=uuid.uuid4().hex,
        request_id=get_trace_id() or uuid.uuid4().hex,
    )
