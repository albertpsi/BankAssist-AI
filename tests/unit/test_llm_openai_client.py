"""OpenAI adapter, exercised entirely against a fake SDK (FR-L1-6, AC-L1-17).

No test here constructs a real OpenAI client or opens a socket.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError

from bankassist.config import Settings
from bankassist.errors import LLMError
from bankassist.llm.base import LLMMessage
from bankassist.llm.openai_client import OpenAIClient
from bankassist.tracing.span import SpanStatus, SpanType
from bankassist.tracing.tracer import InMemoryTracer


def _completion(
    content: str = "an answer",
    *,
    model: str = "test-fast-model",
    prompt_tokens: int = 12,
    completion_tokens: int = 7,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Mimic the shape the SDK returns, without importing its response types."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeCompletions:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _client_with(settings: Settings, result: Any, tracer: InMemoryTracer | None = None):
    """Build an OpenAIClient whose SDK handle is a fake."""
    client = OpenAIClient(settings, tracer)
    fake = FakeCompletions(result)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # noqa: SLF001
    return client, fake


def test_successful_call_maps_onto_our_response(settings: Settings) -> None:
    client, _ = _client_with(settings, _completion())

    response = client.complete([LLMMessage(role="user", content="what is APR?")])

    assert response.text == "an answer"
    assert response.model == "test-fast-model"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0.0


def test_tier_selects_the_configured_model(settings: Settings) -> None:
    """Callers ask for a tier; the adapter resolves the model id."""
    configured = settings.model_copy(update={"llm_model_strong": "a-stronger-model"})
    client, fake = _client_with(configured, _completion())

    client.complete([LLMMessage(role="user", content="q")], tier="strong")

    assert fake.kwargs["model"] == "a-stronger-model"


def test_max_tokens_override_is_passed_through(settings: Settings) -> None:
    client, fake = _client_with(settings, _completion())

    client.complete([LLMMessage(role="user", content="q")], max_tokens=64)

    assert fake.kwargs["max_completion_tokens"] == 64


def test_configured_max_tokens_used_by_default(settings: Settings) -> None:
    client, fake = _client_with(settings, _completion())

    client.complete([LLMMessage(role="user", content="q")])

    assert fake.kwargs["max_completion_tokens"] == settings.llm_max_tokens


def test_messages_are_serialized_as_role_content(settings: Settings) -> None:
    client, fake = _client_with(settings, _completion())

    client.complete(
        [
            LLMMessage(role="system", content="be helpful"),
            LLMMessage(role="user", content="hello"),
        ]
    )

    assert fake.kwargs["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]


def test_provider_errors_are_wrapped(settings: Settings) -> None:
    """AC-L1-12: SDK exception types must not escape the llm package."""
    sdk_error = APIConnectionError(request=None)  # type: ignore[arg-type]
    client, _ = _client_with(settings, sdk_error)

    with pytest.raises(LLMError) as excinfo:
        client.complete([LLMMessage(role="user", content="q")])

    assert excinfo.value.details["provider"] == "openai"
    assert excinfo.value.__cause__ is sdk_error


def test_wrapped_error_message_does_not_leak_the_api_key(settings: Settings) -> None:
    """The credential must not reach an exception a caller could log."""
    client, _ = _client_with(settings, APIConnectionError(request=None))  # type: ignore[arg-type]

    with pytest.raises(LLMError) as excinfo:
        client.complete([LLMMessage(role="user", content="q")])

    rendered = f"{excinfo.value.message} {excinfo.value.details}"
    assert "sk-test-not-a-real-key" not in rendered


def test_empty_choices_raises_rather_than_returning_blank(settings: Settings) -> None:
    empty = SimpleNamespace(model="m", choices=[], usage=None)
    client, _ = _client_with(settings, empty)

    with pytest.raises(LLMError, match="no choices"):
        client.complete([LLMMessage(role="user", content="q")])


def test_missing_usage_block_does_not_crash(settings: Settings) -> None:
    """Not every model returns usage; a working call should still succeed."""
    no_usage = SimpleNamespace(
        model="m",
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
        usage=None,
    )
    client, _ = _client_with(settings, no_usage)

    response = client.complete([LLMMessage(role="user", content="q")])

    assert response.text == "hi"
    assert response.usage.input_tokens == 0


def test_null_content_becomes_empty_string(settings: Settings) -> None:
    null_content = SimpleNamespace(
        model="m",
        choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="length")],
        usage=None,
    )
    client, _ = _client_with(settings, null_content)

    assert client.complete([LLMMessage(role="user", content="q")]).text == ""


def test_call_emits_an_llm_span(settings: Settings, tracer: InMemoryTracer) -> None:
    """Labs 6 and 7 depend on this span existing from Lab 1 onward."""
    client, _ = _client_with(settings, _completion(), tracer)

    client.complete([LLMMessage(role="user", content="q")])

    (span,) = tracer.spans()
    assert span.type is SpanType.LLM_CALL
    assert span.attributes["model"] == "test-fast-model"
    assert span.attributes["input_tokens"] == 12
    assert span.attributes["output_tokens"] == 7


def test_failed_call_marks_the_span_as_error(settings: Settings, tracer: InMemoryTracer) -> None:
    client, _ = _client_with(settings, APIConnectionError(request=None), tracer)  # type: ignore[arg-type]

    with pytest.raises(LLMError):
        client.complete([LLMMessage(role="user", content="q")])

    (span,) = tracer.spans()
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "LLMError"
