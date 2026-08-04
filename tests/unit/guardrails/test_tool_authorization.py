from bankassist.guardrails.tool_authorization import (
    check_dispute_mutation_allowed,
    check_permission,
)
from bankassist.security.authorize import Permission
from bankassist.security.context import SecurityContext

CUSTOMER = SecurityContext(
    user_id="USR-1", role="CUSTOMER", customer_id="CUST001", session_id="S1", request_id="R1"
)
SUPPORT_AGENT = SecurityContext(
    user_id="USR-2", role="SUPPORT_AGENT", customer_id=None, session_id="S2", request_id="R2"
)


def test_permission_granted_allows():
    result = check_permission(CUSTOMER, Permission.VIEW_OWN_TRANSACTIONS)
    assert result.allowed is True


def test_missing_permission_blocks():
    # SUPPORT_AGENT does not hold CREATE_OWN_DISPUTE (AC-10).
    result = check_permission(SUPPORT_AGENT, Permission.CREATE_OWN_DISPUTE)
    assert result.allowed is False
    assert result.category == "AUTHORIZATION"


def test_own_scope_mismatch_blocks():
    result = check_permission(
        CUSTOMER, Permission.CREATE_OWN_DISPUTE, resource_customer_id="CUST002"
    )
    assert result.allowed is False


# --- create_dispute mutation invariant (Lab 5 §6 / AC-12/13/14) ---


def test_mutation_blocked_when_no_pending_action():
    result = check_dispute_mutation_allowed(
        approval_status=None, pending_action=None, already_consumed=False
    )
    assert result.allowed is False


def test_mutation_blocked_before_approval():
    result = check_dispute_mutation_allowed(
        approval_status=None,
        pending_action={"transaction_id": "TX1"},
        already_consumed=False,
    )
    assert result.allowed is False


def test_mutation_blocked_after_rejection():
    result = check_dispute_mutation_allowed(
        approval_status="rejected",
        pending_action={"transaction_id": "TX1"},
        already_consumed=False,
    )
    assert result.allowed is False


def test_mutation_blocked_on_replay():
    result = check_dispute_mutation_allowed(
        approval_status="approved",
        pending_action={"transaction_id": "TX1"},
        already_consumed=True,
    )
    assert result.allowed is False


def test_mutation_allowed_when_approved_and_not_consumed():
    result = check_dispute_mutation_allowed(
        approval_status="approved",
        pending_action={"transaction_id": "TX1"},
        already_consumed=False,
    )
    assert result.allowed is True
