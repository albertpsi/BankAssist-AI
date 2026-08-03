"""Tracer interface and in-process implementations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

from bankassist.context import get_trace_id
from bankassist.tracing.span import Span, SpanStatus, SpanType

_current_span_id: ContextVar[str | None] = ContextVar("bankassist_current_span", default=None)


class Tracer(Protocol):
    """Records timed, nested spans.

    A ``Protocol`` rather than a base class, so implementations — including test
    doubles — need no inheritance coupling.
    """

    @contextmanager
    def span(self, span_type: SpanType, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a span for the duration of the block."""
        ...

    def spans(self) -> list[Span]:
        """Return the spans recorded so far."""
        ...


class InMemoryTracer:
    """Collects spans in a list. Lab 6 adds persistence and a viewer."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    @contextmanager
    def span(self, span_type: SpanType, name: str, **attributes: Any) -> Iterator[Span]:
        record = Span(
            type=span_type,
            name=name,
            trace_id=get_trace_id(),
            parent_span_id=_current_span_id.get(),
            attributes=dict(attributes),
        )
        token = _current_span_id.set(record.span_id)
        started = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            # Record the failure, then let it propagate. Tracing must never turn a
            # crash into a silent success.
            record.status = SpanStatus.ERROR
            record.error_type = type(exc).__name__
            record.error_message = str(exc)
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000.0
            _current_span_id.reset(token)
            self._spans.append(record)

    def spans(self) -> list[Span]:
        return list(self._spans)


class NoOpTracer:
    """Discards everything. Used when tracing is disabled."""

    @contextmanager
    def span(self, span_type: SpanType, name: str, **attributes: Any) -> Iterator[Span]:
        yield Span(type=span_type, name=name, attributes=dict(attributes))

    def spans(self) -> list[Span]:
        return []


def build_tracer(*, enabled: bool) -> Tracer:
    """Return the tracer implied by configuration."""
    return InMemoryTracer() if enabled else NoOpTracer()
