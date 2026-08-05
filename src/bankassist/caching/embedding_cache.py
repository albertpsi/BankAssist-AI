"""Embedding cache (Lab 7, ADR-0013).

Wraps ``rag.embeddings.OpenAIEmbedder`` without modifying its call shape: the
embedder gains an optional collaborator, `OpenAIEmbedder(settings, embedding_cache=...)`,
and calls `get`/`set_many` around the API call it already makes. Nothing about
`embed_documents`/`embed_query`'s signature, tracing, or error handling changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from bankassist.caching.stats import record as record_stat
from bankassist.caching.stats import record_latency
from bankassist.config import Settings
from bankassist.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """SHA256(model + text) -> embedding vector, in Redis.

    The model identifier is hashed *into* the key (Lab 7 amendment #2) rather
    than only prefixed onto it, so a future embedding model change can never
    collide with, or accidentally reuse, a vector produced by a different
    model — the whole input to the hash changes, not just a separate path
    segment next to it.
    """

    def __init__(self, client: Any | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.hits = 0
        self.misses = 0
        self.last_latency_ms: float | None = None

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._settings.embedding_cache_enabled

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the cached vector for `(model, text)`, or None on a miss."""
        if not self.enabled:
            return None

        key = self._key(model, text)
        started = time.perf_counter()
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning("Embedding cache GET failed; treating as a miss.", exc_info=True)
            raw = None
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        record_latency(self._client, self.last_latency_ms)

        if raw is None:
            self.misses += 1
            record_stat(self._client, "embedding_misses")
            return None

        try:
            vector = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Embedding cache entry was unparseable; treating as a miss.")
            self.misses += 1
            record_stat(self._client, "embedding_misses")
            return None

        self.hits += 1
        record_stat(self._client, "embedding_hits")
        return vector

    def set(self, model: str, text: str, vector: list[float]) -> None:
        """Store `vector` for `(model, text)`, with the configured TTL."""
        if not self.enabled:
            return

        key = self._key(model, text)
        try:
            self._client.set(
                key,
                json.dumps(vector),
                ex=self._settings.embedding_cache_ttl_seconds,
            )
        except Exception:
            logger.warning(
                "Embedding cache SET failed; continuing without caching this entry.",
                exc_info=True,
            )

    def _key(self, model: str, text: str) -> str:
        digest = hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()
        return f"embedding:{self._settings.cache_key_version}:{digest}"
