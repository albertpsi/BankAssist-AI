"""JWT issue/verify (ADR-0010). Local, symmetric-key, demo-appropriate signing.

Claims are minimal: ``sub`` (user id), ``role``, ``customer_id`` (nullable), ``exp``.
No refresh tokens, no revocation list — out of scope by design (Lab 4 brief amendment).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from bankassist.config import Settings
from bankassist.errors import AuthenticationError


def issue_token(*, settings: Settings, user_id: str, role: str, customer_id: str | None) -> str:
    """Sign a short-lived access token for an authenticated user."""
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "customer_id": customer_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expiry_minutes),
    }
    return jwt.encode(
        claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )


def decode_token(token: str, *, settings: Settings) -> dict[str, Any]:
    """Verify signature and expiry, return claims.

    Raises:
        AuthenticationError: missing, malformed, expired, or tampered token. The
            message is deliberately generic (FR-22) — never distinguishes "expired"
            from "invalid signature" to a caller, only in the internal log.
    """
    try:
        return jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid or expired credentials.") from exc
