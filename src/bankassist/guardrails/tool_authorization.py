"""Explicit tool-authorization boundary (Lab 5).

``security.authorize.authorize()`` (Lab 4) already runs *inside* every scoped tool.
This module adds a second, explicit check at the graph-node call site, so that
"the supervisor routed here" is never mistaken for authorization: whichever node the
graph reaches, this function is what actually decides whether the tool call may
proceed, and it produces a traceable ``GuardrailResult``/``ExecutionEvent`` regardless
of which route got the agent there. It wraps, and never replaces, the tool's own
check — defense in depth, not a second source of truth for permissions.

Also owns the ``create_dispute`` mutation invariant (Lab 5 §6): approved, not
rejected, not already consumed.
"""

from __future__ import annotations

from bankassist.errors import AuthorizationError
from bankassist.guardrails.models import GuardrailCategory, GuardrailResult
from bankassist.security.authorize import Permission, authorize
from bankassist.security.context import SecurityContext


def check_permission(
    context: SecurityContext,
    permission: Permission,
    *,
    resource_customer_id: str | None = None,
) -> GuardrailResult:
    """Re-verify permission + ownership before a tool call, independent of routing."""
    try:
        authorize(context, permission, resource_customer_id=resource_customer_id)
    except AuthorizationError as exc:
        return GuardrailResult.block(
            f"tool_authorization.{permission.value.lower()}",
            GuardrailCategory.AUTHORIZATION,
            reason="You don't have permission to perform this action.",
            internal_reason=exc.message,
            metadata={"permission": permission.value},
        )
    return GuardrailResult.allow(
        f"tool_authorization.{permission.value.lower()}",
        GuardrailCategory.AUTHORIZATION,
        reason=f"Authorized: {permission.value}.",
    )


def check_dispute_mutation_allowed(
    *,
    approval_status: str | None,
    pending_action: dict[str, object] | None,
    already_consumed: bool,
) -> GuardrailResult:
    """The financial-mutation invariant for ``create_dispute`` (Lab 5 §6).

    ``create_dispute`` may run only when a ``pending_action`` exists, it was
    explicitly approved (never merely "not rejected"), and this specific approval has
    not already been consumed once (replay protection).
    """
    if pending_action is None:
        return GuardrailResult.block(
            "tool_authorization.dispute_mutation",
            GuardrailCategory.TOOL,
            reason="No dispute action is pending approval.",
            internal_reason="no_pending_action",
        )
    if already_consumed:
        return GuardrailResult.block(
            "tool_authorization.dispute_mutation",
            GuardrailCategory.TOOL,
            reason="This approval has already been used.",
            internal_reason="approval_replay_blocked",
        )
    if approval_status != "approved":
        return GuardrailResult.block(
            "tool_authorization.dispute_mutation",
            GuardrailCategory.TOOL,
            reason="This action requires human approval before it can proceed.",
            internal_reason=f"approval_status={approval_status!r}",
        )
    return GuardrailResult.allow(
        "tool_authorization.dispute_mutation",
        GuardrailCategory.TOOL,
        reason="Approved and not previously consumed.",
    )
