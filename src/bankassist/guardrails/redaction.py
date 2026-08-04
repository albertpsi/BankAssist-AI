"""Deterministic secret redaction (Lab 5).

One reusable ``redact()`` instead of scattered ``.replace()`` calls (per instruction).
Applied at logging, exception-to-response boundaries, and as the final pass on any
agent output before it leaves the system. Pattern-based, not semantic — this is the
control that must hold even if NeMo is unavailable or wrong.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization: Bearer <token>
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_.=]+"), "Bearer [REDACTED]"),
    # JWT — three dot-separated base64url segments.
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "[REDACTED_JWT]"),
    # OpenAI-shaped API keys.
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
    # Pinecone-shaped API keys (UUID-like).
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b(?=.*(?i:pinecone|api.?key))"
        ),
        "[REDACTED_API_KEY]",
    ),
    # bcrypt hash.
    (re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}"), "[REDACTED_HASH]"),
]

_FIELD_NAMES = ("password_hash", "jwt_secret", "api_key", "authorization", "access_token")


def redact(text: str) -> str:
    """Replace any known secret-shaped substring in ``text``. Never raises."""
    if not text:
        return text
    redacted = text
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_mapping(data: dict[str, object]) -> dict[str, object]:
    """Redact known-sensitive field names in a shallow dict (for log/event payloads)."""
    result: dict[str, object] = {}
    for key, value in data.items():
        if key.lower() in _FIELD_NAMES:
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            result[key] = redact(value)
        else:
            result[key] = value
    return result
