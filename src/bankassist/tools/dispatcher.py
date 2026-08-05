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
from typing import Any

from bankassist.caching.tool_cache import ToolCache
from bankassist.errors import BankAssistError
from bankassist.execution_event import (
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
)
from bankassist.observability import run as observability_run
from bankassist.observability import update_metadata


def call_tool[T](
    *,
    node_id: str,
    label: str,
    fn: Callable[[], T],
    tool_cache: ToolCache | None = None,
    cache_args: dict[str, Any] | None = None,
) -> tuple[T | None, list[ExecutionEvent]]:
    """Run ``fn`` (a zero-arg closure over one scoped tool call), recording events.

    Returns ``(result, events)``. On failure (including an ``AuthorizationError``
    raised inside the tool), ``result`` is ``None`` and the failure event's summary
    never contains the denied resource's data — only the error's own safe message.

    ``tool_cache``/``cache_args`` are optional (Lab 7, ADR-0013): when given, a
    deterministic tool result is looked up before ``fn`` runs and stored after.
    On a cache hit, ``fn`` is never called — so any ``authorize()``/ownership
    check written inside ``fn`` does **not** run for that call. Safety here
    comes entirely from ``ToolCache``'s own positive allowlist
    (``TOOL_CACHE_ALLOWLIST``): it refuses to serve or store anything for a
    label not on that list, and today that list contains no customer-scoped
    tool. Do not pass ``tool_cache``/``cache_args`` for a tool whose
    authorization depends on per-call arguments (e.g. ``customer_id``) unless
    that tool's label is deliberately kept off the allowlist.
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

    if tool_cache is not None and cache_args is not None:
        cached = tool_cache.get(label, cache_args)
        _emit_tool_cache_event(events, node_id, label, hit=cached is not None)
        if cached is not None:
            return cached, events  # type: ignore[return-value]

    started = time.perf_counter()
    try:
        # AgentOps' automatic OpenAI/LangGraph instrumentation doesn't see a
        # plain Python function call — this is the one deliberate custom span
        # per tool invocation (Lab 6 requirements §5). No-ops when AgentOps
        # is disabled (the default, and always the case under test).
        result = observability_run("tool", label, fn)
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
    if tool_cache is not None and cache_args is not None:
        tool_cache.set(label, cache_args, result)
    return result, events


def _emit_tool_cache_event(
    events: list[ExecutionEvent], node_id: str, label: str, *, hit: bool
) -> None:
    """Record the tool-cache decision — hit or miss, never silently — both in
    the ``ExecutionEvent`` timeline and, when AgentOps is enabled, on the
    active trace (Lab 7 amendment #5: record hits *and* misses, not just hits).
    """
    events.append(
        ExecutionEvent(
            event_type=ExecutionEventType.TOOL_CACHE_HIT
            if hit
            else ExecutionEventType.TOOL_CACHE_MISS,
            node_id=f"{node_id}_tool_cache",
            node_type="cache",
            label=f"Tool Cache: {label}",
            status=ExecutionStatus.COMPLETED,
            summary=f"{label} tool cache {'hit' if hit else 'miss'}",
        )
    )
    update_metadata(tool_cache_event="hit" if hit else "miss", tool_cache_label=label)
