"""``_transaction_options`` (Lab 4 UX fix): structured choices for the dispute
"which transaction?" step, so the UI can render buttons instead of parsing free text.
"""

from __future__ import annotations

from bankassist.api.routes.agent import _transaction_options

_TXN = {
    "transaction_id": "TX1007",
    "account_id": "ACC-1",
    "card_id": None,
    "amount_paise": 450000,
    "currency": "INR",
    "merchant": "QuickPay",
    "category": "Shopping",
    "txn_date": "2026-08-01",
    "status": "POSTED",
}


def test_offers_choices_when_dispute_agent_asked_and_none_selected_yet():
    result = {
        "current_agent": "dispute_agent",
        "selected_transaction_id": None,
        "tool_results": {"recent_transactions": [_TXN]},
    }
    options = _transaction_options(result)
    assert options is not None
    assert options[0].transaction_id == "TX1007"
    assert options[0].amount_rupees == 4500.0


def test_none_once_a_transaction_is_selected():
    result = {
        "current_agent": "dispute_agent",
        "selected_transaction_id": "TX1007",
        "tool_results": {"recent_transactions": [_TXN]},
    }
    assert _transaction_options(result) is None


def test_none_for_non_dispute_agents():
    result = {
        "current_agent": "banking_agent",
        "selected_transaction_id": None,
        "tool_results": {"recent_transactions": [_TXN]},
    }
    assert _transaction_options(result) is None


def test_none_when_no_transactions_offered():
    result = {"current_agent": "dispute_agent", "selected_transaction_id": None, "tool_results": {}}
    assert _transaction_options(result) is None
