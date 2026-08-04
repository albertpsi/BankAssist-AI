"""Scoped banking/dispute tools. No generic SQL tool exists (Lab 4, FR-9)."""

from __future__ import annotations

from bankassist.tools.dispatcher import call_tool
from bankassist.tools.scoped_tools import (
    TransactionNotFoundError,
    check_dispute_eligibility,
    create_dispute,
    get_customer_accounts,
    get_recent_transactions,
    get_transaction_details,
)

__all__ = [
    "TransactionNotFoundError",
    "call_tool",
    "check_dispute_eligibility",
    "create_dispute",
    "get_customer_accounts",
    "get_recent_transactions",
    "get_transaction_details",
]
