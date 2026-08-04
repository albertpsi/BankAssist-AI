"""Embeddings adapter, exercised entirely against a fake SDK (FR-L2-4, NFR-L2-2).

No test here opens a socket or spends money.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError

from bankassist.config import Settings
from bankassist.errors import EmbeddingError
from bankassist.rag.embeddings import OpenAIEmbedder
from bankassist.rag.stubs import StubEmbedder
from bankassist.tracing.span import SpanStatus, SpanType
from bankassist.tracing.tracer import InMemoryTracer


class FakeEmbeddings:
    """Stands in for `client.embeddings`, recording what it was asked for."""

    def __init__(self, result: Any = None) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        if self._result is not None:
            return self._result
        # Default: one vector per input, in order, tagged with its index.
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index)] * 4)
                for index in range(len(inputs))
            ]
        )


def _embedder(settings: Settings, result: Any = None, tracer: InMemoryTracer | None = None):
    embedder = OpenAIEmbedder(settings, tracer)
    fake = FakeEmbeddings(result)
    embedder._client = SimpleNamespace(embeddings=fake)  # noqa: SLF001
    return embedder, fake


def test_configured_model_is_sent(settings: Settings) -> None:
    """FR-L2-4.1: the model id is configuration, never a literal at a call site."""
    configured = settings.model_copy(update={"embedding_model": "text-embedding-3-small"})
    embedder, fake = _embedder(configured)

    embedder.embed_documents(["a chunk"])

    assert fake.calls[0]["model"] == "text-embedding-3-small"


def test_documents_are_batched_to_the_configured_size(settings: Settings) -> None:
    """FR-L2-4.2: 250 chunks at batch size 100 is three calls, not 250."""
    configured = settings.model_copy(update={"embedding_batch_size": 100})
    embedder, fake = _embedder(configured)

    vectors = embedder.embed_documents([f"chunk {index}" for index in range(250)])

    assert [len(call["input"]) for call in fake.calls] == [100, 100, 50]
    assert len(vectors) == 250


def test_empty_input_makes_no_api_call(settings: Settings) -> None:
    embedder, fake = _embedder(settings)

    assert embedder.embed_documents([]) == []
    assert fake.calls == []


def test_query_embedding_is_a_single_vector(settings: Settings) -> None:
    embedder, fake = _embedder(settings)

    vector = embedder.embed_query("what is the chargeback time limit?")

    assert vector == [0.0, 0.0, 0.0, 0.0]
    assert fake.calls[0]["input"] == ["what is the chargeback time limit?"]


def test_out_of_order_results_are_restored_to_input_order(settings: Settings) -> None:
    """A mis-paired vector would corrupt every later retrieval, undetectably."""
    shuffled = SimpleNamespace(
        data=[
            SimpleNamespace(index=2, embedding=[2.0]),
            SimpleNamespace(index=0, embedding=[0.0]),
            SimpleNamespace(index=1, embedding=[1.0]),
        ]
    )
    embedder, _ = _embedder(settings, shuffled)

    vectors = embedder.embed_documents(["first", "second", "third"])

    assert vectors == [[0.0], [1.0], [2.0]]


def test_short_response_is_rejected_rather_than_silently_mispaired(settings: Settings) -> None:
    truncated = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])
    embedder, _ = _embedder(settings, truncated)

    with pytest.raises(EmbeddingError, match="1 embeddings for 3 inputs"):
        embedder.embed_documents(["a", "b", "c"])


def test_provider_errors_are_wrapped(settings: Settings) -> None:
    """AC: SDK exception types must not escape the rag package."""
    sdk_error = APIConnectionError(request=None)  # type: ignore[arg-type]
    embedder, _ = _embedder(settings, sdk_error)

    with pytest.raises(EmbeddingError) as excinfo:
        embedder.embed_documents(["a chunk"])

    assert excinfo.value.details["provider"] == "openai"
    assert excinfo.value.__cause__ is sdk_error


def test_wrapped_error_does_not_leak_the_api_key(settings: Settings) -> None:
    embedder, _ = _embedder(settings, APIConnectionError(request=None))  # type: ignore[arg-type]

    with pytest.raises(EmbeddingError) as excinfo:
        embedder.embed_query("q")

    rendered = f"{excinfo.value.message} {excinfo.value.details}"
    assert "sk-test-not-a-real-key" not in rendered


def test_document_embedding_emits_a_span(settings: Settings, tracer: InMemoryTracer) -> None:
    embedder, _ = _embedder(settings, tracer=tracer)

    embedder.embed_documents(["a", "b"])

    (span,) = tracer.spans()
    assert span.type is SpanType.EMBEDDING
    assert span.attributes["model"] == settings.embedding_model
    assert span.attributes["input_count"] == 2
    assert span.attributes["vector_count"] == 2


def test_failed_embedding_marks_the_span_as_error(
    settings: Settings, tracer: InMemoryTracer
) -> None:
    embedder, _ = _embedder(settings, APIConnectionError(request=None), tracer)  # type: ignore[arg-type]

    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])

    (span,) = tracer.spans()
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "EmbeddingError"


class TestStubEmbedder:
    """The double every other RAG test depends on."""

    def test_is_deterministic(self) -> None:
        stub = StubEmbedder()

        assert stub.embed_query("chargeback") == stub.embed_query("chargeback")

    def test_different_text_gives_a_different_vector(self) -> None:
        stub = StubEmbedder()

        assert stub.embed_query("chargeback") != stub.embed_query("kyc")

    def test_respects_the_configured_dimensions(self) -> None:
        stub = StubEmbedder(dimensions=1536)

        assert len(stub.embed_query("q")) == 1536
        assert len(stub.embed_documents(["a", "b"])[0]) == 1536

    def test_records_what_it_was_asked_to_embed(self) -> None:
        stub = StubEmbedder(dimensions=8)

        stub.embed_documents(["a", "b"])
        stub.embed_query("q")

        assert stub.embedded_documents == [["a", "b"]]
        assert stub.embedded_queries == ["q"]
