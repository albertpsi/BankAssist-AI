"""Password hashing (ADR-0010). Passwords are never stored or logged in plaintext.

Uses the ``bcrypt`` package directly rather than ``passlib``'s bcrypt wrapper, which
is incompatible with ``bcrypt`` 4.1+'s removed ``__about__`` module.
"""

from __future__ import annotations

import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plaintext password for storage."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    """Check a plaintext password against a stored hash. Never raises on mismatch."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed/foreign hash format — treat as a non-match, not a crash.
        return False
