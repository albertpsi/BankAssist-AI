"""Request-scoped correlation context.

A single source of truth for the current trace id, shared by the logger and the
tracer so that a log line and a span can be joined after the fact. Kept in its own
leaf module to avoid a logging ↔ tracing import cycle.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_trace_id: ContextVar[str | None] = ContextVar("bankassist_trace_id", default=None)


def new_trace_id() -> str:
    """Generate an id for a fresh trace."""
    return uuid.uuid4().hex


def get_trace_id() -> str | None:
    """Return the trace id for the current context, if one is set."""
    return _trace_id.get()


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace id for the duration of the block, restoring the previous one after."""
    resolved = trace_id or new_trace_id()
    token = _trace_id.set(resolved)
    try:
        yield resolved
    finally:
        _trace_id.reset(token)
