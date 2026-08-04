"""Centralized RBAC (ADR-0010, FR-26/FR-27/FR-30).

Permissions are declared once, here, and checked by the tool dispatcher before any
tool body runs. Nothing about authorization is decided by the LLM: ``authorize()`` is
plain deterministic code, independently unit-testable without a model.
"""

from __future__ import annotations

from enum import StrEnum

from bankassist.errors import AuthorizationError
from bankassist.security.context import SecurityContext


class Role(StrEnum):
    CUSTOMER = "CUSTOMER"
    SUPPORT_AGENT = "SUPPORT_AGENT"
    ADMIN = "ADMIN"


class Permission(StrEnum):
    VIEW_OWN_ACCOUNTS = "VIEW_OWN_ACCOUNTS"
    VIEW_OWN_TRANSACTIONS = "VIEW_OWN_TRANSACTIONS"
    VIEW_CUSTOMER_DATA = "VIEW_CUSTOMER_DATA"
    CHECK_DISPUTE_ELIGIBILITY = "CHECK_DISPUTE_ELIGIBILITY"
    CREATE_OWN_DISPUTE = "CREATE_OWN_DISPUTE"
    INVESTIGATE_DISPUTE = "INVESTIGATE_DISPUTE"
    ADMIN_ACCESS = "ADMIN_ACCESS"


# FR-30: the authoritative role -> permission matrix. Extend here, not per-call-site.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.CUSTOMER: frozenset(
        {
            Permission.VIEW_OWN_ACCOUNTS,
            Permission.VIEW_OWN_TRANSACTIONS,
            Permission.CHECK_DISPUTE_ELIGIBILITY,
            Permission.CREATE_OWN_DISPUTE,
        }
    ),
    Role.SUPPORT_AGENT: frozenset(
        {
            Permission.VIEW_CUSTOMER_DATA,
            Permission.INVESTIGATE_DISPUTE,
            Permission.CHECK_DISPUTE_ELIGIBILITY,
        }
    ),
    Role.ADMIN: frozenset(
        {
            Permission.VIEW_OWN_ACCOUNTS,
            Permission.VIEW_OWN_TRANSACTIONS,
            Permission.VIEW_CUSTOMER_DATA,
            Permission.CHECK_DISPUTE_ELIGIBILITY,
            Permission.CREATE_OWN_DISPUTE,
            Permission.INVESTIGATE_DISPUTE,
            Permission.ADMIN_ACCESS,
        }
    ),
}


def permissions_for_role(role: str) -> frozenset[Permission]:
    try:
        return ROLE_PERMISSIONS[Role(role)]
    except ValueError:
        return frozenset()


def authorize(
    context: SecurityContext,
    permission: Permission,
    *,
    resource_customer_id: str | None = None,
) -> None:
    """Deny unless ``context`` has ``permission`` and, for own-scope permissions,
    ``resource_customer_id`` (if given) matches ``context.customer_id``.

    Raises:
        AuthorizationError: permission missing, or an own-scope permission is being
            used to reach a resource that belongs to a different customer.
    """
    granted = permissions_for_role(context.role)
    if permission not in granted:
        raise AuthorizationError(
            "This action requires a permission the current role does not have.",
            details={"permission": permission.value, "role": context.role},
        )

    own_scope_permissions = {
        Permission.VIEW_OWN_ACCOUNTS,
        Permission.VIEW_OWN_TRANSACTIONS,
        Permission.CREATE_OWN_DISPUTE,
    }
    if (
        permission in own_scope_permissions
        and resource_customer_id is not None
        and resource_customer_id != context.customer_id
    ):
        raise AuthorizationError(
            "This action can only be performed on the caller's own customer data.",
            details={"permission": permission.value},
        )
