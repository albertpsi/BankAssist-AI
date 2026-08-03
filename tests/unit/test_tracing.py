"""Span model and tracer (FR-L1-7)."""

from __future__ import annotations

import pytest

from bankassist.context import trace_context
from bankassist.tracing.span import SpanStatus, SpanType
from bankassist.tracing.tracer import InMemoryTracer, NoOpTracer, build_tracer


def test_span_records_type_name_and_duration(tracer: InMemoryTracer) -> None:
    with tracer.span(SpanType.REQUEST, "GET /health"):
        pass

    (span,) = tracer.spans()
    assert span.type is SpanType.REQUEST
    assert span.name == "GET /health"
    assert span.status is SpanStatus.OK
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0


def test_nested_spans_record_parent_ids(tracer: InMemoryTracer) -> None:
    """AC-L1-13: nesting must be reconstructable from the flat span list."""
    with tracer.span(SpanType.REQUEST, "outer"):
        with tracer.span(SpanType.LLM_CALL, "inner"):
            pass

    inner, outer = tracer.spans()  # children close first
    assert inner.name == "inner"
    assert outer.name == "outer"
    assert inner.parent_span_id == outer.span_id
    assert outer.parent_span_id is None


def test_sibling_spans_share_a_parent(tracer: InMemoryTracer) -> None:
    with tracer.span(SpanType.REQUEST, "outer"):
        with tracer.span(SpanType.LLM_CALL, "first"):
            pass
        with tracer.span(SpanType.LLM_CALL, "second"):
            pass

    first, second, outer = tracer.spans()
    assert first.parent_span_id == outer.span_id
    assert second.parent_span_id == outer.span_id
    assert first.span_id != second.span_id


def test_trace_id_propagates_from_context(tracer: InMemoryTracer) -> None:
    with trace_context("trace-abc"), tracer.span(SpanType.REQUEST, "outer"):
        with tracer.span(SpanType.LLM_CALL, "inner"):
            pass

    assert {span.trace_id for span in tracer.spans()} == {"trace-abc"}


def test_span_records_error_and_reraises(tracer: InMemoryTracer) -> None:
    """AC-L1-14: tracing must never convert a crash into a silent success."""
    with pytest.raises(ValueError, match="kaboom"):
        with tracer.span(SpanType.LLM_CALL, "failing"):
            raise ValueError("kaboom")

    (span,) = tracer.spans()
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"
    assert span.error_message == "kaboom"
    assert span.duration_ms is not None


def test_context_unwinds_after_an_error(tracer: InMemoryTracer) -> None:
    """A failed child must not leave itself as the ambient parent."""
    with tracer.span(SpanType.REQUEST, "outer"):
        with pytest.raises(ValueError):
            with tracer.span(SpanType.LLM_CALL, "failing"):
                raise ValueError("boom")
        with tracer.span(SpanType.LLM_CALL, "sibling"):
            pass

    spans = {span.name: span for span in tracer.spans()}
    assert spans["sibling"].parent_span_id == spans["outer"].span_id


def test_attributes_are_recorded(tracer: InMemoryTracer) -> None:
    with tracer.span(SpanType.LLM_CALL, "call", model="test-model") as span:
        span.set_attribute("input_tokens", 42)

    (span_record,) = tracer.spans()
    assert span_record.attributes == {"model": "test-model", "input_tokens": 42}


def test_spans_returns_a_copy(tracer: InMemoryTracer) -> None:
    with tracer.span(SpanType.REQUEST, "one"):
        pass

    tracer.spans().clear()

    assert len(tracer.spans()) == 1


def test_noop_tracer_records_nothing() -> None:
    """AC-L1-15."""
    noop = NoOpTracer()

    with noop.span(SpanType.REQUEST, "ignored") as span:
        span.set_attribute("key", "value")

    assert noop.spans() == []


def test_build_tracer_honours_the_flag() -> None:
    assert isinstance(build_tracer(enabled=True), InMemoryTracer)
    assert isinstance(build_tracer(enabled=False), NoOpTracer)
