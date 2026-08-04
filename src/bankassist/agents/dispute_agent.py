"""Dispute Agent (Lab 4, §5). Banking tools + Enterprise RAG + Dispute tools.

Pure helper functions the graph node composes — kept separate from LangGraph so each
piece (amount extraction, transaction resolution) is independently unit-testable
without a graph or a live LLM.
"""

from __future__ import annotations

import re

from bankassist.tools.models import Transaction

_AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr)?\s?([\d][\d,]*(?:\.\d{1,2})?)", re.IGNORECASE)


def extract_amount_rupees(text: str) -> float | None:
    """Pull the first rupee-shaped amount out of free text, e.g. "the ₹4,500 one".

    Deterministic parsing, not an LLM guess (Lab 4 brief §11): multi-turn resolution
    must be reproducible.
    """
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def resolve_transaction_id(
    message: str, candidates: list[Transaction], *, tolerance_rupees: float = 1.0
) -> str | None:
    """Match a follow-up message like "the ₹4,500 one" to a previously shown transaction.

    Resolves purely against transactions already surfaced in this session's state
    (``candidates``) — never a fresh unscoped lookup — matching the multi-turn
    requirement in Lab 4 brief §11.
    """
    amount = extract_amount_rupees(message)
    if amount is None:
        return None
    for txn in candidates:
        if abs(txn.amount_paise / 100 - amount) <= tolerance_rupees:
            return txn.transaction_id
    return None


DISPUTE_POLICY_QUESTION = (
    "What is the process and eligibility window for disputing an unrecognized "
    "credit or debit card transaction?"
)
