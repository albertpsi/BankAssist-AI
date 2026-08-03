"""Structured logging (FR-L1-3)."""

from __future__ import annotations

import json
import logging

import pytest

from bankassist.context import trace_context
from bankassist.logging_config import JsonFormatter, configure_logging, get_logger


def _record(**kwargs: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="bankassist.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def test_record_is_valid_json_with_core_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "bankassist.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload
    assert "module" in payload


def test_extra_fields_are_emitted() -> None:
    payload = json.loads(JsonFormatter().format(_record(status_code=200, path="/health")))

    assert payload["status_code"] == 200
    assert payload["path"] == "/health"


def test_ambient_trace_id_is_promoted_to_top_level() -> None:
    with trace_context("abc123"):
        payload = json.loads(JsonFormatter().format(_record()))

    assert payload["trace_id"] == "abc123"


def test_explicit_trace_id_wins_over_ambient() -> None:
    with trace_context("ambient"):
        payload = json.loads(JsonFormatter().format(_record(trace_id="explicit")))

    assert payload["trace_id"] == "explicit"


def test_trace_id_absent_when_none_is_set() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    assert "trace_id" not in payload


def test_exception_info_is_captured() -> None:
    try:
        raise ValueError("kaboom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: kaboom" in payload["exception"]


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO", force=True)
    handler_count = len(logging.getLogger().handlers)

    configure_logging("INFO")
    configure_logging("INFO")

    assert len(logging.getLogger().handlers) == handler_count


def test_configured_output_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", force=True)

    get_logger("bankassist.test").info("structured", extra={"widget": "value"})

    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["widget"] == "value"
