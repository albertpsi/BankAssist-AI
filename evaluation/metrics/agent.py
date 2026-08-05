"""Agent-level correctness: routing, tool selection, approval gating, mutation
timing (Lab 6 requirements §14).
"""

from __future__ import annotations


def routing_correct(actual_route: str | None, expected_route: str | None) -> bool:
    if expected_route is None:
        return True
    return (actual_route or "").upper() == expected_route.upper()


def tools_match(
    tools_called: list[str], expected_tools: list[str], unexpected_tools: list[str]
) -> bool:
    """All expected tools were called, and no explicitly-unexpected tool was."""
    called = set(tools_called)
    expected_ok = set(expected_tools).issubset(called)
    unexpected_ok = called.isdisjoint(set(unexpected_tools))
    return expected_ok and unexpected_ok


def approval_gate_correct(actual_required: bool, expected_required: bool | None) -> bool:
    if expected_required is None:
        return True
    return actual_required == expected_required


def mutation_never_precedes_approval(mutation_occurred_before_approval: bool) -> bool:
    """The one invariant that must hold on every case, dispute or not."""
    return mutation_occurred_before_approval is False
