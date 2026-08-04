from pathlib import Path

import pytest

from bankassist.errors import AuthorizationError
from bankassist.security.context import SecurityContext
from bankassist.tools import (
    TransactionNotFoundError,
    check_dispute_eligibility,
    create_dispute,
    get_customer_accounts,
    get_recent_transactions,
    get_transaction_details,
)


def _ctx(customer_id: str) -> SecurityContext:
    return SecurityContext(
        user_id="USR-1", role="CUSTOMER", customer_id=customer_id, session_id="S1", request_id="R1"
    )


def test_get_customer_accounts_returns_only_own_accounts(db_path: Path):
    result = get_customer_accounts(_ctx("CUST001"), db_path)
    assert [a.account_id for a in result.accounts] == ["ACC-1"]


def test_get_recent_transactions_scoped_to_customer(db_path: Path):
    result = get_recent_transactions(_ctx("CUST002"), db_path)
    assert [t.transaction_id for t in result.transactions] == ["TX2"]


def test_get_transaction_details_rejects_cross_customer_access(db_path: Path):
    with pytest.raises(AuthorizationError):
        get_transaction_details(_ctx("CUST001"), db_path, transaction_id="TX2")


def test_get_transaction_details_no_data_leaks_on_rejection(db_path: Path):
    try:
        get_transaction_details(_ctx("CUST001"), db_path, transaction_id="TX2")
        pytest.fail("expected AuthorizationError")
    except AuthorizationError as exc:
        assert "BigBasket" not in str(exc.details)
        assert "99999" not in str(exc.details)


def test_get_transaction_details_missing_transaction_raises_not_found(db_path: Path):
    with pytest.raises(TransactionNotFoundError):
        get_transaction_details(_ctx("CUST001"), db_path, transaction_id="TX-NOPE")


def test_llm_supplied_customer_id_mismatch_is_rejected_not_corrected(db_path: Path):
    with pytest.raises(AuthorizationError):
        get_customer_accounts(_ctx("CUST001"), db_path, requested_customer_id="CUST002")


def test_llm_supplied_customer_id_matching_context_is_allowed(db_path: Path):
    result = get_customer_accounts(_ctx("CUST001"), db_path, requested_customer_id="CUST001")
    assert len(result.accounts) == 1


def test_check_dispute_eligibility_eligible_for_fresh_posted_transaction(db_path: Path):
    result = check_dispute_eligibility(_ctx("CUST001"), db_path, transaction_id="TX1")
    assert result.eligible is True


def test_check_dispute_eligibility_cross_customer_rejected(db_path: Path):
    with pytest.raises(AuthorizationError):
        check_dispute_eligibility(_ctx("CUST002"), db_path, transaction_id="TX1")


def test_create_dispute_succeeds_for_own_eligible_transaction(db_path: Path):
    result = create_dispute(_ctx("CUST001"), db_path, transaction_id="TX1", reason="Not recognized")
    assert result.status == "OPEN"
    assert result.reference.startswith("DSP-")


def test_create_dispute_rejects_another_customers_transaction(db_path: Path):
    with pytest.raises(AuthorizationError):
        create_dispute(_ctx("CUST002"), db_path, transaction_id="TX1", reason="Not recognized")


def test_create_dispute_twice_is_not_eligible_second_time(db_path: Path):
    create_dispute(_ctx("CUST001"), db_path, transaction_id="TX1", reason="Not recognized")
    eligibility = check_dispute_eligibility(_ctx("CUST001"), db_path, transaction_id="TX1")
    assert eligibility.eligible is False
