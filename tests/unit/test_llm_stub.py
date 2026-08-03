"""The stub client and the shared LLM types (FR-L1-6)."""

from __future__ import annotations

import pytest

from bankassist.errors import LLMError
from bankassist.llm.base import LLMMessage, LLMResponse, TokenUsage
from bankassist.llm.stub import StubLLMClient


def test_token_usage_totals() -> None:
    assert TokenUsage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_response_carries_accounting_fields() -> None:
    """AC-L1-10: Labs 6 and 7 read these off every call."""
    response = LLMResponse(
        text="hi",
        model="test-model",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        latency_ms=12.5,
        finish_reason="stop",
    )

    assert response.model == "test-model"
    assert response.usage.total_tokens == 15
    assert response.latency_ms == 12.5


def test_stub_returns_queued_responses_in_order() -> None:
    """AC-L1-9."""
    client = StubLLMClient(["first", "second"])
    messages = [LLMMessage(role="user", content="q")]

    assert client.complete(messages).text == "first"
    assert client.complete(messages).text == "second"


def test_stub_accepts_full_response_objects() -> None:
    scripted = LLMResponse(text="crafted", model="m", usage=TokenUsage(input_tokens=99))
    client = StubLLMClient([scripted])

    assert client.complete([LLMMessage(role="user", content="q")]).usage.input_tokens == 99


def test_stub_records_calls_for_assertions() -> None:
    client = StubLLMClient(["ok"])
    messages = [
        LLMMessage(role="system", content="you are a banking assistant"),
        LLMMessage(role="user", content="what is APR?"),
    ]

    client.complete(messages, tier="strong", max_tokens=256)

    call = client.last_call()
    assert call.tier == "strong"
    assert call.max_tokens == 256
    assert [m.role for m in call.messages] == ["system", "user"]
    assert len(client.calls) == 1


def test_stub_raises_when_script_is_exhausted() -> None:
    """A silent empty response would hide a bug in the code under test."""
    client = StubLLMClient(["only one"])
    messages = [LLMMessage(role="user", content="q")]
    client.complete(messages)

    with pytest.raises(LLMError, match="exhausted"):
        client.complete(messages)


def test_last_call_fails_loudly_when_nothing_was_called() -> None:
    with pytest.raises(AssertionError, match="no calls"):
        StubLLMClient().last_call()


def test_stub_reflects_the_requested_tier_in_the_model_name() -> None:
    client = StubLLMClient(["a", "b"])
    messages = [LLMMessage(role="user", content="q")]

    assert client.complete(messages, tier="fast").model == "stub-fast"
    assert client.complete(messages, tier="strong").model == "stub-strong"
