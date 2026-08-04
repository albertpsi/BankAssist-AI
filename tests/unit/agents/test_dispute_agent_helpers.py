from bankassist.agents.dispute_agent import extract_amount_rupees, resolve_transaction_id
from bankassist.tools.models import Transaction


def _txn(txn_id: str, amount_paise: int) -> Transaction:
    return Transaction(
        transaction_id=txn_id,
        account_id="ACC-1",
        card_id=None,
        amount_paise=amount_paise,
        currency="INR",
        merchant="M",
        category="Shopping",
        txn_date="2026-08-01",
        status="POSTED",
    )


def test_extract_amount_rupees_handles_rupee_symbol_and_commas():
    assert extract_amount_rupees("The ₹4,500 one") == 4500.0


def test_extract_amount_rupees_handles_plain_number():
    assert extract_amount_rupees("that 750 charge") == 750.0


def test_extract_amount_rupees_returns_none_when_absent():
    assert extract_amount_rupees("I don't recognize a transaction") is None


def test_resolve_transaction_id_matches_by_amount():
    candidates = [_txn("TX1007", 450000), _txn("TX1002", 45000)]
    assert resolve_transaction_id("The ₹4,500 one", candidates) == "TX1007"


def test_resolve_transaction_id_returns_none_when_no_match():
    candidates = [_txn("TX1007", 450000)]
    assert resolve_transaction_id("The ₹99 one", candidates) is None
