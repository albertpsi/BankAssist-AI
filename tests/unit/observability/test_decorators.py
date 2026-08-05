"""Decorators/`run()` must be pure pass-throughs whenever AgentOps is disabled
— the state the whole test suite otherwise runs in (Lab 6 §21)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from bankassist.observability import agent_span, operation, run, tool_span, trace, workflow
from bankassist.observability.agentops_client import reset_for_tests

F = TypeVar("F", bound=Callable[..., object])


@pytest.fixture(autouse=True)
def _ensure_disabled():
    reset_for_tests()
    yield
    reset_for_tests()


def test_operation_decorator_passthrough() -> None:
    @operation("test.op")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_tool_span_decorator_passthrough() -> None:
    @tool_span("test.tool", cost=0.01)
    def echo(x: str) -> str:
        return x

    assert echo("hello") == "hello"


def test_agent_and_workflow_and_trace_passthrough() -> None:
    @agent_span("test.agent")
    def a() -> str:
        return "a"

    @workflow("test.workflow")
    def w() -> str:
        return "w"

    @trace("test.trace")
    def t() -> str:
        return "t"

    assert (a(), w(), t()) == ("a", "w", "t")


def test_run_helper_forwards_args_and_kwargs() -> None:
    def fn(a: int, *, b: int) -> int:
        return a * b

    assert run("operation", "test.run", fn, 4, b=5) == 20


def test_run_helper_propagates_exceptions() -> None:
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run("operation", "test.boom", boom)


def test_wrapped_function_is_never_called_twice_when_enabled_and_it_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a broad `except Exception` around the call itself (not
    just around acquiring the AgentOps decorator) would silently re-invoke a
    mutating function a second time after it raised — a double-mutation bug
    for e.g. `create_dispute`."""
    import sys

    class _FakeDecorators:
        @staticmethod
        def operation(**_kwargs: object) -> Callable[[F], F]:
            def deco(fn: F) -> F:
                return fn

            return deco

    fake_module = type(sys)("agentops.sdk")
    fake_module.decorators = _FakeDecorators()
    monkeypatch.setitem(sys.modules, "agentops.sdk", fake_module)
    monkeypatch.setattr("bankassist.observability.agentops_client.is_enabled", lambda: True)

    calls: list[int] = []

    def mutate() -> None:
        calls.append(1)
        raise ValueError("mutation failed")

    with pytest.raises(ValueError, match="mutation failed"):
        run("operation", "test.mutate", mutate)

    assert len(calls) == 1
