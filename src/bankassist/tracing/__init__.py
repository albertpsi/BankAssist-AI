"""Tracing foundation.

The span model and tracer interface exist from Lab 1 so that every layer calls the
tracer from its first line of code. Labs 6 and 7 read these spans; retrofitting
them later was identified as a cross-cutting rework risk in the implementation plan.

Persistence is deliberately out of scope until Lab 6.
"""

from bankassist.tracing.span import Span, SpanStatus, SpanType
from bankassist.tracing.tracer import InMemoryTracer, NoOpTracer, Tracer

__all__ = [
    "InMemoryTracer",
    "NoOpTracer",
    "Span",
    "SpanStatus",
    "SpanType",
    "Tracer",
]
