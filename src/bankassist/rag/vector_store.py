"""Pinecone vector store (ADR-0007).

The only module in the application that imports the Pinecone SDK. Its exceptions
are wrapped in ``VectorStoreError`` here so they never leak past the ``rag``
package, the same contract ``llm/openai_client.py`` holds for OpenAI.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from bankassist.config import Settings
from bankassist.errors import ConfigurationError, VectorStoreError
from bankassist.logging_config import get_logger
from bankassist.rag.models import DocumentMetadata, RetrievedChunk, VectorRecord
from bankassist.tracing.span import SpanType
from bankassist.tracing.tracer import NoOpTracer, Tracer

logger = get_logger(__name__)

# A newly created serverless index is not immediately queryable.
_READY_TIMEOUT_SECONDS = 120.0
_READY_POLL_SECONDS = 2.0

# Pinecone caps a single upsert request; batching also gives us per-batch logging.
_UPSERT_BATCH_SIZE = 100


class VectorStore(Protocol):
    """Stores embedded chunks and finds the nearest ones to a query vector."""

    def ensure_index(self) -> bool:
        """Create the index if it is absent. Returns True when it created one."""
        ...

    def upsert(self, records: list[VectorRecord]) -> int:
        """Write records, overwriting any with the same id. Returns the count."""
        ...

    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks, best first."""
        ...

    def count(self) -> int:
        """How many vectors the namespace currently holds."""
        ...


class PineconeVectorStore:
    """Implements ``VectorStore`` against a Pinecone serverless index."""

    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        # Settings deliberately does not enforce this key at startup — /health and
        # the test suite must work on a machine with no Pinecone account. So the
        # check lands here, at the first point it actually matters.
        if not settings.has_pinecone_credential():
            raise ConfigurationError(
                "PINECONE_API_KEY is required to use the vector store, and must not "
                "be blank. Set it in .env.",
                details={"field": "pinecone_api_key"},
            )

        self._settings = settings
        self._tracer = tracer or NoOpTracer()
        self._client = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())  # type: ignore[union-attr]
        self._index: Any | None = None

    @property
    def namespace(self) -> str:
        return self._settings.pinecone_namespace

    @property
    def index_name(self) -> str:
        return self._settings.pinecone_index_name

    def ensure_index(self) -> bool:
        """Create the index if it is missing, and wait until it can serve."""
        try:
            if self._client.has_index(self.index_name):
                return False

            logger.info(
                "creating pinecone index",
                extra={
                    "index": self.index_name,
                    "dimension": self._settings.embedding_dimensions,
                    "metric": "cosine",
                    "cloud": self._settings.pinecone_cloud,
                    "region": self._settings.pinecone_region,
                },
            )
            self._client.create_index(
                name=self.index_name,
                dimension=self._settings.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
        except PineconeException as exc:
            raise self._wrap(exc, "create index") from exc

        self._await_ready()
        return True

    def upsert(self, records: list[VectorRecord]) -> int:
        """Write every record, in batches, overwriting matching ids."""
        if not records:
            return 0

        batches = [
            records[index : index + _UPSERT_BATCH_SIZE]
            for index in range(0, len(records), _UPSERT_BATCH_SIZE)
        ]

        started = time.perf_counter()
        written = 0
        for number, batch in enumerate(batches, start=1):
            payload = [
                {"id": record.id, "values": record.values, "metadata": record.metadata}
                for record in batch
            ]
            try:
                self._handle().upsert(vectors=payload, namespace=self.namespace)
            except PineconeException as exc:
                raise self._wrap(exc, "upsert") from exc

            written += len(batch)
            logger.info(
                "pinecone upsert batch complete",
                extra={
                    "index": self.index_name,
                    "namespace": self.namespace,
                    "batch": number,
                    "batch_count": len(batches),
                    "vectors": len(batch),
                },
            )

        logger.info(
            "pinecone upsert complete",
            extra={
                "index": self.index_name,
                "namespace": self.namespace,
                "vectors_upserted": written,
                "batch_count": len(batches),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
        )
        return written

    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Plain vector similarity. No filter, no hybrid, no rerank — that is Lab 3."""
        span_ctx = self._tracer.span(
            SpanType.RETRIEVAL,
            "pinecone.query",
            namespace=self.namespace,
            top_k=top_k,
        )
        started = time.perf_counter()
        with span_ctx as span:
            try:
                response = self._handle().query(
                    vector=vector,
                    top_k=top_k,
                    namespace=self.namespace,
                    include_metadata=True,
                )
            except PineconeException as exc:
                raise self._wrap(exc, "query") from exc

            chunks = [_to_chunk(match) for match in _matches(response)]
            span.set_attribute("result_count", len(chunks))
            span.set_attribute("top_score", chunks[0].score if chunks else None)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "retrieval complete",
            extra={
                "namespace": self.namespace,
                "top_k": top_k,
                "result_count": len(chunks),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        for rank, chunk in enumerate(chunks, start=1):
            logger.info(
                "retrieval result",
                extra={
                    "rank": rank,
                    "document": chunk.metadata.document,
                    "chunk_index": chunk.chunk_index,
                    "score": round(chunk.score, 6),
                },
            )
        return chunks

    def count(self) -> int:
        """Vectors in this namespace. Zero when the namespace does not exist yet."""
        try:
            stats = self._handle().describe_index_stats()
        except PineconeException as exc:
            raise self._wrap(exc, "describe index stats") from exc

        namespaces = _as_dict(stats).get("namespaces") or {}
        entry = _as_dict(namespaces.get(self.namespace, {}))
        return int(entry.get("vector_count", 0))

    def _handle(self) -> Any:
        """The index client, opened once and reused."""
        if self._index is None:
            try:
                self._index = self._client.Index(self.index_name)
            except PineconeException as exc:
                raise self._wrap(exc, "open index") from exc
        return self._index

    def _await_ready(self) -> None:
        """Block until a freshly created serverless index can serve queries."""
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                description = self._client.describe_index(self.index_name)
            except PineconeException as exc:
                raise self._wrap(exc, "describe index") from exc

            status = _as_dict(_as_dict(description).get("status", {}))
            if status.get("ready"):
                logger.info("pinecone index ready", extra={"index": self.index_name})
                return
            time.sleep(_READY_POLL_SECONDS)

        raise VectorStoreError(
            f"Pinecone index {self.index_name!r} was not ready within "
            f"{_READY_TIMEOUT_SECONDS:.0f}s of being created.",
            details={"index": self.index_name},
        )

    def _wrap(self, exc: PineconeException, operation: str) -> VectorStoreError:
        """Name the operation and the type, never the request or the credential."""
        return VectorStoreError(
            f"Pinecone {operation} failed: {type(exc).__name__}",
            details={"index": self.index_name, "namespace": self.namespace},
        )


def _matches(response: Any) -> list[Any]:
    """Read the match list off a query response, which may be object or dict."""
    if isinstance(response, dict):
        return list(response.get("matches") or [])
    return list(getattr(response, "matches", None) or [])


def _to_chunk(match: Any) -> RetrievedChunk:
    """Rebuild a chunk from what the store gave back."""
    payload = _as_dict(match)
    metadata = _as_dict(payload.get("metadata") or {})

    return RetrievedChunk(
        text=str(metadata.get("text", "")),
        metadata=DocumentMetadata(
            document=str(metadata.get("document", "")),
            title=str(metadata.get("title", "")),
            category=str(metadata.get("category", "")),
            source=str(metadata.get("source", "")),
        ),
        chunk_index=int(float(metadata.get("chunk_index", 0))),
        score=float(payload.get("score", 0.0)),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalize an SDK response object into a plain dict.

    The SDK returns objects that are dict-like in some versions and attribute-only
    in others; reading defensively here keeps a minor SDK release from breaking
    ingestion.
    """
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {}
