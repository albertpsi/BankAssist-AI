"""Retrieval metrics, evaluated separately from generation (Lab 6 requirements §12).

All three operate on ``retrieved`` (a ranked list of source document names,
best first) versus ``expected`` (the golden set of acceptable source
documents for the query) — no LLM call, fully deterministic.
"""

from __future__ import annotations


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> bool:
    """True if any expected source appears in the top ``k`` retrieved sources."""
    if not expected:
        raise ValueError("hit_at_k requires at least one expected source.")
    return bool(set(retrieved[:k]) & set(expected))


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected sources present in the top ``k`` retrieved sources."""
    if not expected:
        raise ValueError("recall_at_k requires at least one expected source.")
    found = set(retrieved[:k]) & set(expected)
    return len(found) / len(expected)


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """Reciprocal rank of the first expected source found in ``retrieved``.

    0.0 when no expected source appears at all.
    """
    if not expected:
        raise ValueError("mrr requires at least one expected source.")
    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0
