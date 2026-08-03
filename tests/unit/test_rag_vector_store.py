"""Pinecone adapter, exercised entirely against a fake SDK client (FR-L2-5, FR-L2-6).

Mirrors the pattern in test_llm_openai_client.py and test_rag_embeddings.py: no
test here opens a socket, and none needs a Pinecone account (NFR-L2-2).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pinecone.exceptions import PineconeException
from pydantic import SecretStr

from bankassist.config import Settings
from bankassist.errors import ConfigurationError, VectorStoreError
from bankassist.rag.models import DocumentMetadata, VectorRecord
from bankassist.rag.vector_store import PineconeVectorStore
from bankassist.tracing.span import SpanStatus, SpanType
from bankassist.tracing.tracer import InMemoryTracer


class FakeIndexHandle:
    """Stands in for `Pinecone().Index(name)`."""

    def __init__(self, query_result: Any = None, stats_result: Any = None) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self._query_result = query_result
        self._stats_result = stats_result

    def upsert(self, **kwargs: Any) -> Any:
        if isinstance(self._query_result, PineconeException) and kwargs.get("_raise"):
            raise self._query_result
        self.upsert_calls.append(kwargs)
        return SimpleNamespace(upserted_count=len(kwargs["vectors"]))

    def query(self, **kwargs: Any) -> Any:
        self.query_calls.append(kwargs)
        if isinstance(self._query_result, Exception):
            raise self._query_result
        return self._query_result

    def describe_index_stats(self, **kwargs: Any) -> Any:
        if isinstance(self._stats_result, Exception):
            raise self._stats_result
        return self._stats_result


class FakeClient:
    """Stands in for the top-level `Pinecone(api_key=...)` object."""

    def __init__(
        self,
        *,
        has_index: bool = True,
        index_handle: FakeIndexHandle | None = None,
        ready_after: int = 0,
    ) -> None:
        self._has_index = has_index
        self._index_handle = index_handle or FakeIndexHandle()
        self.create_index_calls: list[dict[str, Any]] = []
        self._describe_calls = 0
        self._ready_after = ready_after

    def has_index(self, name: str) -> bool:
        return self._has_index

    def create_index(self, **kwargs: Any) -> Any:
        self.create_index_calls.append(kwargs)
        self._has_index = True
        return SimpleNamespace(name=kwargs["name"])

    def describe_index(self, name: str) -> Any:
        self._describe_calls += 1
        ready = self._describe_calls > self._ready_after
        return {"status": {"ready": ready}}

    def Index(self, name: str) -> FakeIndexHandle:  # noqa: N802 (matches SDK's method name)
        return self._index_handle


def _record(
    document: str = "a.md", index: int = 0, values: list[float] | None = None
) -> VectorRecord:
    return VectorRecord(
        id=f"{document}#{index}",
        values=values or [0.1, 0.2, 0.3],
        metadata={
            "document": document,
            "title": document,
            "category": "Credit Card",
            "source": "Test Source",
            "chunk_index": index,
            "text": "the chargeback window is 90 days",
        },
    )


def _store(
    settings: Settings,
    client: FakeClient | None = None,
    tracer: InMemoryTracer | None = None,
    index_handle: FakeIndexHandle | None = None,
) -> tuple[PineconeVectorStore, FakeClient]:
    configured = settings.model_copy(
        update={"pinecone_api_key": SecretStr("pc-test-not-a-real-key")}
    )
    store = PineconeVectorStore(configured, tracer)
    fake = client or FakeClient(index_handle=index_handle)
    store._client = fake  # noqa: SLF001
    return store, fake


def test_missing_credential_is_rejected_at_construction(settings: Settings) -> None:
    """FR-L2-9.4: the failure must be here, not a confusing SDK 401 later."""
    with pytest.raises(ConfigurationError) as excinfo:
        PineconeVectorStore(settings)

    assert excinfo.value.details["field"] == "pinecone_api_key"


def test_blank_credential_is_rejected(settings: Settings) -> None:
    configured = settings.model_copy(update={"pinecone_api_key": SecretStr("   ")})

    with pytest.raises(ConfigurationError):
        PineconeVectorStore(configured)


class TestEnsureIndex:
    def test_existing_index_is_not_recreated(self, settings: Settings) -> None:
        store, client = _store(settings, FakeClient(has_index=True))

        created = store.ensure_index()

        assert created is False
        assert client.create_index_calls == []

    def test_missing_index_is_created_with_the_configured_dimension(
        self, settings: Settings
    ) -> None:
        configured = settings.model_copy(update={"embedding_dimensions": 1536})
        store, client = _store(configured, FakeClient(has_index=False))

        created = store.ensure_index()

        assert created is True
        assert client.create_index_calls[0]["dimension"] == 1536
        assert client.create_index_calls[0]["metric"] == "cosine"

    def test_waits_for_the_new_index_to_report_ready(self, settings: Settings) -> None:
        store, client = _store(settings, FakeClient(has_index=False, ready_after=2))

        store.ensure_index()

        assert client._describe_calls > 2  # noqa: SLF001

    def test_sdk_error_is_wrapped(self, settings: Settings) -> None:
        class BoomClient(FakeClient):
            def has_index(self, name: str) -> bool:
                raise PineconeException("boom")

        store, _ = _store(settings, BoomClient())

        with pytest.raises(VectorStoreError) as excinfo:
            store.ensure_index()

        assert "create index" in excinfo.value.message


class TestUpsert:
    def test_empty_input_makes_no_call(self, settings: Settings) -> None:
        store, client = _store(settings)

        assert store.upsert([]) == 0
        assert client._index_handle.upsert_calls == []  # noqa: SLF001

    def test_all_records_are_written_in_the_configured_namespace(
        self, settings: Settings
    ) -> None:
        configured = settings.model_copy(update={"pinecone_namespace": "bank-policies"})
        store, client = _store(configured)

        written = store.upsert([_record("a.md", 0), _record("a.md", 1)])

        assert written == 2
        assert client._index_handle.upsert_calls[0]["namespace"] == "bank-policies"  # noqa: SLF001

    def test_upsert_batches_large_inputs(self, settings: Settings) -> None:
        store, client = _store(settings)
        records = [_record("a.md", i) for i in range(150)]

        store.upsert(records)

        calls = client._index_handle.upsert_calls  # noqa: SLF001
        assert [len(c["vectors"]) for c in calls] == [100, 50]

    def test_record_fields_map_onto_the_pinecone_payload(self, settings: Settings) -> None:
        store, client = _store(settings)

        store.upsert([_record("chargeback.md", 3, values=[0.5, 0.6])])

        payload = client._index_handle.upsert_calls[0]["vectors"][0]  # noqa: SLF001
        assert payload["id"] == "chargeback.md#3"
        assert payload["values"] == [0.5, 0.6]
        assert payload["metadata"]["document"] == "chargeback.md"


class TestQuery:
    def _response(self, matches: list[dict[str, Any]]) -> dict[str, Any]:
        return {"matches": matches}

    def _match(
        self, document: str = "a.md", chunk_index: int = 0, score: float = 0.9
    ) -> dict[str, Any]:
        return {
            "score": score,
            "metadata": {
                "document": document,
                "title": document,
                "category": "Credit Card",
                "source": "Test Source",
                "chunk_index": chunk_index,
                "text": "the chargeback window is 90 days",
            },
        }

    def test_matches_are_mapped_onto_retrieved_chunks(self, settings: Settings) -> None:
        match = self._match("chargeback.md", 2, 0.87)
        handle = FakeIndexHandle(query_result=self._response([match]))
        store, _ = _store(settings, index_handle=handle)

        (chunk,) = store.query([0.1, 0.2], top_k=5)

        assert chunk.metadata == DocumentMetadata(
            document="chargeback.md", title="chargeback.md", category="Credit Card",
            source="Test Source",
        )
        assert chunk.chunk_index == 2
        assert chunk.score == 0.87
        assert chunk.text == "the chargeback window is 90 days"

    def test_top_k_and_namespace_are_passed_through(self, settings: Settings) -> None:
        configured = settings.model_copy(update={"pinecone_namespace": "bank-policies"})
        handle = FakeIndexHandle(query_result=self._response([]))
        store, _ = _store(configured, index_handle=handle)

        store.query([0.1], top_k=7)

        assert handle.query_calls[0]["top_k"] == 7
        assert handle.query_calls[0]["namespace"] == "bank-policies"

    def test_object_shaped_response_is_also_handled(self, settings: Settings) -> None:
        """The SDK's return type is object-attribute-based in some versions."""
        response = SimpleNamespace(
            matches=[SimpleNamespace(score=0.5, metadata={"document": "b.md", "chunk_index": 0})]
        )
        handle = FakeIndexHandle(query_result=response)
        store, _ = _store(settings, index_handle=handle)

        (chunk,) = store.query([0.1], top_k=1)

        assert chunk.metadata.document == "b.md"

    def test_no_matches_returns_an_empty_list(self, settings: Settings) -> None:
        handle = FakeIndexHandle(query_result=self._response([]))
        store, _ = _store(settings, index_handle=handle)

        assert store.query([0.1], top_k=5) == []

    def test_sdk_error_is_wrapped_and_does_not_leak_the_key(self, settings: Settings) -> None:
        handle = FakeIndexHandle(query_result=PineconeException("boom"))
        store, _ = _store(settings, index_handle=handle)

        with pytest.raises(VectorStoreError) as excinfo:
            store.query([0.1], top_k=5)

        rendered = f"{excinfo.value.message} {excinfo.value.details}"
        assert "pc-test-not-a-real-key" not in rendered

    def test_query_emits_a_retrieval_span(self, settings: Settings) -> None:
        tracer = InMemoryTracer()
        handle = FakeIndexHandle(query_result=self._response([self._match(score=0.75)]))
        store, _ = _store(settings, tracer=tracer, index_handle=handle)

        store.query([0.1], top_k=5)

        (span,) = tracer.spans()
        assert span.type is SpanType.RETRIEVAL
        assert span.attributes["result_count"] == 1
        assert span.attributes["top_score"] == 0.75

    def test_failed_query_marks_the_span_as_error(self, settings: Settings) -> None:
        tracer = InMemoryTracer()
        handle = FakeIndexHandle(query_result=PineconeException("boom"))
        store, _ = _store(settings, tracer=tracer, index_handle=handle)

        with pytest.raises(VectorStoreError):
            store.query([0.1], top_k=5)

        (span,) = tracer.spans()
        assert span.status is SpanStatus.ERROR


class TestCount:
    def test_reads_the_configured_namespace_vector_count(self, settings: Settings) -> None:
        configured = settings.model_copy(update={"pinecone_namespace": "bank-policies"})
        handle = FakeIndexHandle(
            stats_result={"namespaces": {"bank-policies": {"vector_count": 190}}}
        )
        store, _ = _store(configured, index_handle=handle)

        assert store.count() == 190

    def test_missing_namespace_counts_as_zero(self, settings: Settings) -> None:
        handle = FakeIndexHandle(stats_result={"namespaces": {}})
        store, _ = _store(settings, index_handle=handle)

        assert store.count() == 0

    def test_sdk_error_is_wrapped(self, settings: Settings) -> None:
        handle = FakeIndexHandle(stats_result=PineconeException("boom"))
        store, _ = _store(settings, index_handle=handle)

        with pytest.raises(VectorStoreError):
            store.count()
