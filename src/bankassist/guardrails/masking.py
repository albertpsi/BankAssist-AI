"""Deterministic PII/financial-data masking (Lab 5, CLAUDE.md §7).

The seeded dataset already stores card numbers pre-masked (``****-****-****-4321``,
see ``scripts/seed_banking_data.py``) and ``account_id``/``customer_id`` are opaque
synthetic labels, not real account numbers — neither needs masking to be *displayed*.

What this module guards against is a raw, unmasked identifier appearing in an agent's
free-text output despite that (e.g. a policy document quoting an example PAN, or a
future regression that stops masking at the source). CLAUDE.md §7 requires exactly
this: "Output guardrails scan responses for unmasked card numbers, SSN-shaped
strings, and full account numbers before the response leaves the system."

Never touches ``amount``, ``merchant``, ``date``, or ``status`` fields — this masks
identifier-shaped substrings only, never rewrites banking facts.
"""

from __future__ import annotations

import re

# 13-19 digits, optionally grouped by spaces/dashes — a card-PAN shape.
_CARD_NUMBER = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# SSN-shaped: NNN-NN-NNNN.
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

MASKED_CARD_FORMAT = re.compile(r"^\*{4}-\*{4}-\*{4}-\d{4}$")


def _mask_digits(match: re.Match[str]) -> str:
    digits = re.sub(r"[ -]", "", match.group(0))
    if len(digits) < 13:
        return match.group(0)
    return f"{'*' * (len(digits) - 4)}{digits[-4:]}"


def mask_sensitive_identifiers(text: str) -> str:
    """Mask any unmasked card-PAN-shaped or SSN-shaped substring in ``text``."""
    if not text:
        return text
    masked = _CARD_NUMBER.sub(_mask_digits, text)
    return _SSN.sub("***-**-****", masked)


def is_masked_card(value: str) -> bool:
    """True if ``value`` already matches the repo's masked-card display format."""
    return bool(MASKED_CARD_FORMAT.match(value))
