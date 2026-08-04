import pytest

from bankassist.errors import AuthorizationError
from bankassist.security.authorize import Permission, authorize
from bankassist.security.context import SecurityContext


def _ctx(role: str, customer_id: str | None) -> SecurityContext:
    return SecurityContext(
        user_id="USR-1", role=role, customer_id=customer_id, session_id="S1", request_id="R1"
    )


def test_customer_can_view_own_accounts():
    authorize(
        _ctx("CUSTOMER", "CUST001"), Permission.VIEW_OWN_ACCOUNTS, resource_customer_id="CUST001"
    )


def test_customer_cannot_view_another_customers_accounts():
    with pytest.raises(AuthorizationError):
        authorize(
            _ctx("CUSTOMER", "CUST001"),
            Permission.VIEW_OWN_ACCOUNTS,
            resource_customer_id="CUST002",
        )


def test_customer_lacks_admin_access():
    with pytest.raises(AuthorizationError):
        authorize(_ctx("CUSTOMER", "CUST001"), Permission.ADMIN_ACCESS)


def test_support_agent_gets_investigate_but_not_admin():
    authorize(_ctx("SUPPORT_AGENT", None), Permission.INVESTIGATE_DISPUTE)
    with pytest.raises(AuthorizationError):
        authorize(_ctx("SUPPORT_AGENT", None), Permission.ADMIN_ACCESS)


def test_admin_has_full_matrix():
    for permission in Permission:
        authorize(_ctx("ADMIN", None), permission)


def test_unknown_role_has_no_permissions():
    with pytest.raises(AuthorizationError):
        authorize(_ctx("GHOST", None), Permission.VIEW_OWN_ACCOUNTS)
