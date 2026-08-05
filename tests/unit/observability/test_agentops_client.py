"""AgentOps init must stay off unless explicitly enabled and configured, and
must never take down the app if the SDK itself misbehaves (Lab 6 §21)."""

from __future__ import annotations

import pytest

from bankassist.config import Settings
from bankassist.observability import agentops_client


@pytest.fixture(autouse=True)
def _reset_agentops_state():
    agentops_client.reset_for_tests()
    yield
    agentops_client.reset_for_tests()


def test_disabled_by_default(settings: Settings) -> None:
    assert settings.agentops_enabled is False
    assert agentops_client.init_agentops(settings) is False
    assert agentops_client.is_enabled() is False


def test_enabled_without_key_stays_off(settings: Settings) -> None:
    enabled = settings.model_copy(update={"agentops_enabled": True, "agentops_api_key": None})
    assert agentops_client.init_agentops(enabled) is False
    assert agentops_client.is_enabled() is False


def test_enabled_with_blank_key_stays_off(settings: Settings) -> None:
    from pydantic import SecretStr

    enabled = settings.model_copy(
        update={"agentops_enabled": True, "agentops_api_key": SecretStr("   ")}
    )
    assert agentops_client.init_agentops(enabled) is False


def test_sdk_failure_is_swallowed(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real key but an unreachable/broken SDK must not raise into the caller."""
    from pydantic import SecretStr

    enabled = settings.model_copy(
        update={"agentops_enabled": True, "agentops_api_key": SecretStr("ao-test-key")}
    )

    class _BrokenAgentops:
        @staticmethod
        def init(**_kwargs: object) -> None:
            raise RuntimeError("network unreachable")

    monkeypatch.setitem(__import__("sys").modules, "agentops", _BrokenAgentops())
    assert agentops_client.init_agentops(enabled) is False
    assert agentops_client.is_enabled() is False


def test_init_excludes_langgraph_from_agentops_auto_instrumentation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: agentops==0.4.21's LangGraph auto-instrumentation replaces
    every graph node with a wrapper that only accepts `(state)`, dropping the
    `config` parameter every BankAssist node relies on — confirmed against a
    running instance of this app (TypeError: ...unexpected keyword argument
    'config'). A prior fix that called `uninstrument()` *after* `init()` did
    not hold — AgentOps' import hook silently re-instruments LangGraph on the
    next relevant import — so `init_agentops` must remove LangGraph from
    AgentOps' target registries *before* `agentops.init()` runs instead."""
    from pydantic import SecretStr

    enabled = settings.model_copy(
        update={"agentops_enabled": True, "agentops_api_key": SecretStr("ao-test-key")}
    )

    init_calls: list[int] = []

    class _FakeAgentops:
        @staticmethod
        def init(**_kwargs: object) -> None:
            init_calls.append(1)

    class _FakeInstrumentationModule:
        AGENTIC_LIBRARIES = {"langgraph": {"class_name": "LanggraphInstrumentor"}, "crewai": {}}
        TARGET_PACKAGES = {"langgraph", "crewai", "openai"}

    import sys

    monkeypatch.setitem(sys.modules, "agentops", _FakeAgentops())
    monkeypatch.setitem(sys.modules, "agentops.instrumentation", _FakeInstrumentationModule())

    assert agentops_client.init_agentops(enabled) is True
    assert len(init_calls) == 1
    assert "langgraph" not in _FakeInstrumentationModule.AGENTIC_LIBRARIES
    assert "langgraph" not in _FakeInstrumentationModule.TARGET_PACKAGES
    # Unaffected libraries stay registered — this is a targeted exclusion.
    assert "crewai" in _FakeInstrumentationModule.AGENTIC_LIBRARIES
    assert "openai" in _FakeInstrumentationModule.TARGET_PACKAGES


def test_second_call_is_a_noop_once_initialized(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import SecretStr

    enabled = settings.model_copy(
        update={"agentops_enabled": True, "agentops_api_key": SecretStr("ao-test-key")}
    )

    calls: list[int] = []

    class _FakeAgentops:
        @staticmethod
        def init(**_kwargs: object) -> None:
            calls.append(1)

    monkeypatch.setitem(__import__("sys").modules, "agentops", _FakeAgentops())
    assert agentops_client.init_agentops(enabled) is True
    assert agentops_client.init_agentops(enabled) is True
    assert len(calls) == 1
