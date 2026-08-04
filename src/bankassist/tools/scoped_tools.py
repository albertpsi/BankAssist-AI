"""The five scoped banking/dispute tools (Lab 4, FR-9/FR-10/FR-25).

Every function here does exactly one thing, has typed input/output, and enforces
customer ownership using ``SecurityContext.customer_id`` — never a caller- or
LLM-supplied ``customer_id``. If an LLM tool-call argument disagrees with the
security context, the call is rejected outright (fail closed), not silently
corrected (FR-25, ADR-0010). There is no generic SQL tool: each function issues one
narrow, parameterized query.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from bankassist.errors import AuthorizationError, BankAssistError
from bankassist.security.authorize import Permission, authorize
from bankassist.security.context import SecurityContext
from bankassist.tools import banking_data
from bankassist.tools.models import (
    Account,
    AccountsResult,
    DisputeResult,
    EligibilityResult,
    Transaction,
    TransactionDetailResult,
    TransactionsResult,
)


class TransactionNotFoundError(BankAssistError):
    code = "transaction_not_found"
    http_status = 404


def _reject_customer_mismatch(context: SecurityContext, requested_customer_id: str | None) -> None:
    """Fail closed when a tool argument disagrees with the security context.

    ``requested_customer_id`` models what an LLM tool call might supply. It is never
    used to select data — the context's ``customer_id`` always is — but a mismatch is
    treated as a rejected request rather than silently overridden (FR-25).
    """
    if requested_customer_id is not None and requested_customer_id != context.customer_id:
        raise AuthorizationError(
            "The requested customer does not match the authenticated identity.",
            details={"permission": "customer_scope"},
        )


def get_customer_accounts(
    context: SecurityContext,
    db_path: Path,
    *,
    requested_customer_id: str | None = None,
) -> AccountsResult:
    """List the caller's own accounts (FR-9)."""
    authorize(context, Permission.VIEW_OWN_ACCOUNTS, resource_customer_id=requested_customer_id)
    _reject_customer_mismatch(context, requested_customer_id)

    with banking_data.session(db_path) as conn:
        rows = conn.execute(
            "SELECT account_id, account_type, balance_paise, currency "
            "FROM accounts WHERE customer_id = ?",
            (context.customer_id,),
        ).fetchall()

    return AccountsResult(accounts=[Account(**dict(row)) for row in rows])


def get_recent_transactions(
    context: SecurityContext,
    db_path: Path,
    *,
    limit: int = 10,
    requested_customer_id: str | None = None,
) -> TransactionsResult:
    """List the caller's most recent transactions, newest first (FR-9)."""
    authorize(context, Permission.VIEW_OWN_TRANSACTIONS, resource_customer_id=requested_customer_id)
    _reject_customer_mismatch(context, requested_customer_id)

    with banking_data.session(db_path) as conn:
        rows = conn.execute(
            "SELECT transaction_id, account_id, card_id, amount_paise, currency, "
            "merchant, category, txn_date, status FROM transactions "
            "WHERE customer_id = ? ORDER BY txn_date DESC, transaction_id DESC LIMIT ?",
            (context.customer_id, limit),
        ).fetchall()

    return TransactionsResult(transactions=[Transaction(**dict(row)) for row in rows])


def get_transaction_details(
    context: SecurityContext,
    db_path: Path,
    *,
    transaction_id: str,
) -> TransactionDetailResult:
    """Fetch one transaction, only if it belongs to the caller (FR-9/FR-10).

    Raises:
        AuthorizationError: the transaction exists but belongs to another customer.
            No data about it is returned.
        TransactionNotFoundError: no such transaction at all.
    """
    authorize(context, Permission.VIEW_OWN_TRANSACTIONS)

    with banking_data.session(db_path) as conn:
        row = conn.execute(
            "SELECT transaction_id, customer_id, account_id, card_id, amount_paise, "
            "currency, merchant, category, txn_date, status FROM transactions "
            "WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()

    if row is None:
        raise TransactionNotFoundError(f"No transaction {transaction_id}.")
    if row["customer_id"] != context.customer_id:
        raise AuthorizationError(
            "This transaction does not belong to the authenticated customer.",
            details={"permission": Permission.VIEW_OWN_TRANSACTIONS.value},
        )

    data = dict(row)
    data.pop("customer_id")
    return TransactionDetailResult(transaction=Transaction(**data))


def check_dispute_eligibility(
    context: SecurityContext,
    db_path: Path,
    *,
    transaction_id: str,
) -> EligibilityResult:
    """Determine whether a transaction can be disputed (FR-9/FR-10).

    Deterministic business rule, no LLM involved: eligible unless the transaction is
    already disputed, or is older than the policy's dispute window. The dispute
    window itself is read from policy via Enterprise RAG at the agent layer, not
    here — this function only applies the mechanical eligibility checks that do not
    require semantic reasoning (already-disputed, ownership).
    """
    authorize(context, Permission.CHECK_DISPUTE_ELIGIBILITY)

    detail = get_transaction_details(context, db_path, transaction_id=transaction_id)

    with banking_data.session(db_path) as conn:
        existing = conn.execute(
            "SELECT 1 FROM disputes WHERE transaction_id = ? AND status != 'REJECTED'",
            (transaction_id,),
        ).fetchone()

    if existing is not None:
        return EligibilityResult(
            transaction_id=transaction_id,
            eligible=False,
            reason="A dispute already exists for this transaction.",
        )
    if detail.transaction.status != "POSTED":
        return EligibilityResult(
            transaction_id=transaction_id,
            eligible=False,
            reason=f"Transaction status '{detail.transaction.status}' is not disputable.",
        )
    return EligibilityResult(transaction_id=transaction_id, eligible=True, reason="Eligible.")


def create_dispute(
    context: SecurityContext,
    db_path: Path,
    *,
    transaction_id: str,
    reason: str,
) -> DisputeResult:
    """Create a dispute case (FR-9). The only write tool — see ADR-0009/ADR-0010:

    the caller must hold ``CREATE_OWN_DISPUTE`` **and** this must only ever be
    reached after the LangGraph human-approval interrupt has been resolved
    affirmatively. Neither check substitutes for the other (FR-31).
    """
    authorize(context, Permission.CREATE_OWN_DISPUTE)

    eligibility = check_dispute_eligibility(context, db_path, transaction_id=transaction_id)
    if not eligibility.eligible:
        raise BankAssistError(
            f"Transaction {transaction_id} is not eligible for dispute: {eligibility.reason}",
            details={"transaction_id": transaction_id},
        )

    dispute_id = f"DSP-{uuid.uuid4().hex[:8].upper()}"
    reference = f"DSP-{datetime.now(UTC).year}-{uuid.uuid4().hex[:6].upper()}"
    created_at = datetime.now(UTC).isoformat()

    with banking_data.session(db_path) as conn:
        conn.execute(
            "INSERT INTO disputes (dispute_id, customer_id, transaction_id, reason, "
            "status, reference, created_at) VALUES (?, ?, ?, ?, 'OPEN', ?, ?)",
            (dispute_id, context.customer_id, transaction_id, reason, reference, created_at),
        )
        conn.commit()

    return DisputeResult(
        dispute_id=dispute_id, reference=reference, transaction_id=transaction_id, status="OPEN"
    )
