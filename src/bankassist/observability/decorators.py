"""Thin, test-safe wrappers around AgentOps' span decorators.

Call sites (``agents/graph.py``, ``rag/pipeline/enterprise_pipeline.py``,
``guardrails/nemo_adapter.py``, ``tools/dispatcher.py``, ``api/routes/agent.py``)
import from here, never from ``agentops`` directly — that keeps AgentOps an
implementation detail of this one package (ADR-0012).

Each wrapper checks ``agentops_client.is_enabled()`` on every call, not at
decoration time: module-level ``@operation(...)`` decorations run at import
time, before ``init_agentops()`` has necessarily been called, so the check
has to be deferred to the actual call. When disabled (the default, and always
true in the unit/integration test suite — CLAUDE.md §21), every wrapper is a
pure pass-through: no AgentOps import, no network, identical return value.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from bankassist.observability import agentops_client

F = TypeVar("F", bound=Callable[..., Any])


def _wrap(kind: str, name: str, **decorator_kwargs: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not agentops_client.is_enabled():
                return fn(*args, **kwargs)

            # Only *building* the span-wrapped callable is defensive here — an
            # AgentOps SDK problem must never block the banking flow it is
            # observing. Once `target` is resolved, it is called exactly
            # once, outside the try block: `fn` may be a mutating tool call
            # (e.g. `create_dispute`), and catching its own exceptions here
            # would fall through to a second, duplicate call below — a
            # double-mutation bug, not a safety net.
            try:
                from agentops.sdk import decorators as _ao

                span_decorator = getattr(_ao, kind)(name=name, **decorator_kwargs)
                target = span_decorator(fn)
            except Exception:  # pragma: no cover - defensive, never blocks the app
                target = fn

            return target(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def operation(name: str) -> Callable[[F], F]:
    """Wraps a BankAssist-specific operation boundary AgentOps can't infer on
    its own — e.g. a supervisor routing decision or one RAG pipeline stage."""
    return _wrap("operation", name)


def agent_span(name: str) -> Callable[[F], F]:
    """Wraps an agent boundary (policy/banking/dispute agent logic)."""
    return _wrap("agent", name)


def tool_span(name: str, *, cost: float | None = None) -> Callable[[F], F]:
    """Wraps a scoped tool call. ``cost`` is optional and left unset unless a
    real per-call cost is known — BankAssist's tools are local SQLite reads."""
    kwargs: dict[str, Any] = {}
    if cost is not None:
        kwargs["cost"] = cost
    return _wrap("tool", name, **kwargs)


def workflow(name: str) -> Callable[[F], F]:
    """Wraps a multi-step workflow boundary (e.g. the HITL dispute flow)."""
    return _wrap("workflow", name)


def trace(name: str) -> Callable[[F], F]:
    """Wraps a whole-request boundary — one BankAssist chat/resume call."""
    return _wrap("trace", name)


def run(kind: str, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` inside a dynamically named span.

    For call sites where the span name is only known at call time — a tool
    name passed into the dispatcher, a RAG stage looped over generically —
    and a static ``@operation("...")`` decoration is not expressible.
    ``kind`` is one of ``"operation"``, ``"agent"``, ``"tool"``, ``"workflow"``.
    """
    return _wrap(kind, name)(fn)(*args, **kwargs)


def update_metadata(**attributes: Any) -> None:
    """Attach sanitized metadata to the currently running AgentOps trace.

    No-op when AgentOps is disabled. Used for control-flow boundaries that
    are neither an LLM call nor a function call AgentOps can wrap — the HITL
    pause/resume moment (Lab 6 requirements §5/§6/§8).
    """
    if not agentops_client.is_enabled():
        return
    try:
        from agentops import update_trace_metadata

        from bankassist.observability.redaction import sanitize_attributes

        update_trace_metadata(sanitize_attributes(attributes))
    except Exception:  # pragma: no cover - defensive, never blocks the app
        pass
