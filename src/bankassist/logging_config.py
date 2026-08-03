"""Structured JSON logging.

One line of JSON per record, with the active trace id promoted to a top-level
field so logs and traces can be joined once Lab 6 persists traces.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from bankassist.context import get_trace_id

# Attributes present on every stdlib LogRecord. Anything outside this set was put
# there by a caller via `extra=` and is therefore worth emitting.
_RESERVED_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }

        # An explicit trace_id passed via `extra=` wins over the ambient one.
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and key != "trace_id":
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: calling it twice is a no-op unless ``force`` is set, so importing a
    module that configures logging cannot silently duplicate handlers.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Library code uses this rather than ``print``."""
    return logging.getLogger(name)
