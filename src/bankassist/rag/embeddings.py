"""Embeddings via the OpenAI API (ADR-0007).

The second module in the application allowed to import the provider SDK. It is
separate from ``llm/`` rather than folded into ``LLMClient`` because the two have
different call shapes, different failure modes, and different cost accounting —
embeddings are priced on input alone and produce no completion.
"""

from __future__ import annotations

import time
from typing import Protocol

from openai import OpenAI, OpenAIError

from bankassist.caching.embedding_cache import EmbeddingCache
from bankassist.config import Settings
from bankassist.errors import ConfigurationError, EmbeddingError
from bankassist.logging_config import get_logger
from bankassist.tracing.span import SpanType
from bankassist.tracing.tracer import NoOpTracer, Tracer

logger = get_logger(__name__)


class Embedder(Protocol):
    """Turns text into vectors.

    Corpus and query embedding are separate methods because they are separate
    operations to trace, log, and — from Lab 7 — cost. They happen to call the
    same model today; nothing here promises they always will.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts, in the order given."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class OpenAIEmbedder:
    """Implements ``Embedder`` against the OpenAI embeddings API."""

    def __init__(
        self,
        settings: Settings,
        tracer: Tracer | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        if settings.openai_api_key is None:
            raise ConfigurationError(
                "OpenAIEmbedder requires OPENAI_API_KEY to be configured.",
                details={"field": "openai_api_key"},
            )

        self._settings = settings
        self._tracer = tracer or NoOpTracer()
        # Lab 7 (ADR-0013): optional collaborator, default None. When absent, this
        # class's behavior is byte-for-byte what Labs 2-6 shipped — an embedding
        # cache is never required for `OpenAIEmbedder` to work.
        self._embedding_cache = embedding_cache
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed every text, batching to the configured batch size."""
        if not texts:
            return []

        batch_size = self._settings.embedding_batch_size
        batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]

        span_ctx = self._tracer.span(
            SpanType.EMBEDDING,
            "openai.embed_documents",
            model=self.model,
            input_count=len(texts),
        )
        started = time.perf_counter()
        vectors: list[list[float]] = []
        with span_ctx as span:
            for number, batch in enumerate(batches, start=1):
                vectors.extend(self._embed(batch))
                logger.info(
                    "embedding batch complete",
                    extra={
                        "model": self.model,
                        "batch": number,
                        "batch_count": len(batches),
                        "batch_size": len(batch),
                    },
                )
            span.set_attribute("batch_count", len(batches))
            span.set_attribute("vector_count", len(vectors))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "embedding generation complete",
            extra={
                "model": self.model,
                "chunk_count": len(texts),
                "batch_count": len(batches),
                "dimensions": len(vectors[0]) if vectors else 0,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        span_ctx = self._tracer.span(SpanType.EMBEDDING, "openai.embed_query", model=self.model)
        with span_ctx:
            return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """One logical embedding call. Wraps SDK errors and restores the
        requested order.

        Lab 7 (ADR-0013): consults the embedding cache first, per-text. Only
        the texts that miss are sent to OpenAI, in one batched call preserving
        their relative order; the results are then merged back into the full
        output in original position. With no cache configured this reduces to
        exactly what Labs 2-6 did.
        """
        cache = self._embedding_cache
        if cache is None or not cache.enabled:
            return self._embed_via_api(texts)

        results: list[list[float] | None] = []
        misses: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = cache.get(self.model, text)
            results.append(cached)
            if cached is None:
                misses.append((index, text))

        if misses:
            fetched = self._embed_via_api([text for _, text in misses])
            for (index, text), vector in zip(misses, fetched, strict=True):
                results[index] = vector
                cache.set(self.model, text, vector)

        return [vector for vector in results if vector is not None]

    def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """One real OpenAI API call. Never consults the cache itself."""
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except OpenAIError as exc:
            # Name the exception type only — never the request body or the key.
            raise EmbeddingError(
                f"OpenAI embeddings request failed: {type(exc).__name__}",
                details={"model": self.model, "provider": "openai", "batch_size": len(texts)},
            ) from exc

        data = getattr(response, "data", None) or []
        if len(data) != len(texts):
            raise EmbeddingError(
                f"OpenAI returned {len(data)} embeddings for {len(texts)} inputs.",
                details={"model": self.model, "provider": "openai"},
            )

        # The API documents that results may come back out of order, so sort by
        # `index` rather than trusting position — a silently mis-paired vector
        # would corrupt every retrieval afterwards, undetectably.
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        return [list(item.embedding) for item in ordered]
