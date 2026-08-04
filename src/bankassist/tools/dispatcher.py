"""Tool dispatcher: the one place agent code calls a scoped tool through (FR-27).

``LLM requests tool -> dispatcher -> RBAC (inside the tool) -> ownership check
(inside the tool) -> execution`` — the dispatcher's job is purely to standardize
timing and ``ExecutionEvent`` emission around every call, so no agent forgets to
record one. RBAC and ownership are enforced inside each scoped tool (`authorize()`
calls in ``scoped_tools``), not re-implemented here — the dispatcher trusts the tool,
not the other way around, so it cannot become a bypass.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from bankassist.errors import BankAssistError
from bankassist.execution_event import (
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
)


def call_tool[T](
    *,
    node_id: str,
    label: str,
    fn: Callable[[], T],
) -> tuple[T | None, list[ExecutionEvent]]:
    """Run ``fn`` (a zero-arg closure over one scoped tool call), recording events.

    Returns ``(result, events)``. On failure (including an ``AuthorizationError``
    raised inside the tool), ``result`` is ``None`` and the failure event's summary
    never contains the denied resource's data — only the error's own safe message.
    """
    events = [
        ExecutionEvent(
            event_type=ExecutionEventType.TOOL_STARTED,
            node_id=node_id,
            node_type="tool",
            label=label,
            status=ExecutionStatus.RUNNING,
            summary=f"Calling {label}",
        )
    ]
    started = time.perf_counter()
    try:
        result = fn()
    except BankAssistError as exc:
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        events.append(
            ExecutionEvent(
                event_type=ExecutionEventType.TOOL_COMPLETED,
                node_id=node_id,
                node_type="tool",
                label=label,
                status=ExecutionStatus.FAILED,
                summary=f"{label} failed ({duration_ms}ms): {exc.message}",
            )
        )
        return None, events

    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    events.append(
        ExecutionEvent(
            event_type=ExecutionEventType.TOOL_COMPLETED,
            node_id=node_id,
            node_type="tool",
            label=label,
            status=ExecutionStatus.COMPLETED,
            summary=f"{label} completed in {duration_ms}ms",
        )
    )
    return result, events
