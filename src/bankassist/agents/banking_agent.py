"""Banking Agent (Lab 4, §4). Scoped tools only — no direct SQLite access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bankassist.observability import run as observability_run
from bankassist.security.context import SecurityContext
from bankassist.tools import get_customer_accounts, get_recent_transactions
from bankassist.tools.models import Account, Transaction


@dataclass(frozen=True)
class BankingAnswer:
    answer: str
    accounts: list[Account]
    transactions: list[Transaction]


def answer_banking_request(context: SecurityContext, db_path: Path) -> BankingAnswer:
    """Summarize the caller's accounts and recent transactions.

    Deterministic formatting, no LLM call: the data itself is the answer, and
    generating prose around a fixed shape would add cost without adding information
    for this lab's scope.
    """
    # Named AgentOps tool spans (Lab 6 requirements §5) — the ExecutionEvents
    # this module's caller emits stay the BankAssist-facing record; these are
    # the operational trace AgentOps surfaces.
    accounts = observability_run(
        "tool", "get_customer_accounts", lambda: get_customer_accounts(context, db_path)
    ).accounts
    transactions = observability_run(
        "tool",
        "get_recent_transactions",
        lambda: get_recent_transactions(context, db_path, limit=5),
    ).transactions

    lines = ["Here is a summary of your accounts and recent transactions:", ""]
    for account in accounts:
        rupees = account.balance_paise / 100
        lines.append(f"- {account.account_type} account {account.account_id}: ₹{rupees:,.2f}")
    lines.append("")
    lines.append("Recent transactions:")
    for txn in transactions:
        rupees = txn.amount_paise / 100
        lines.append(f"- {txn.txn_date}  {txn.merchant}  ₹{rupees:,.2f}  ({txn.transaction_id})")

    return BankingAnswer(answer="\n".join(lines), accounts=accounts, transactions=transactions)
