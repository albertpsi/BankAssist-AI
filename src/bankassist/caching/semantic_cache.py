"""Semantic response cache (Lab 7, ADR-0013).

Reuses ADR-0006's eligibility rule verbatim via ``caching.eligibility`` — this
module owns *storage and lookup*, not the privacy decision. Nothing here ever
stores or serves a response for a request that
``caching.eligibility.classify_eligibility`` did not mark
``CacheEligibility.GLOBAL_CACHEABLE``.

Similarity search: native Redis vector KNN (RediSearch `FT.SEARCH ... KNN`) is
used whenever the connected Redis exposes the `search` module — required by the
Lab 7 amendment ("avoid implementing semantic search in Python unless a
technical blocker is found and documented"). When RediSearch is not loaded
(e.g. a plain `redis:*` image with no modules), a bounded Python-side cosine
scan over the most recent `semantic_cache_max_candidates` eligible entries is
used instead, and this fallback is logged loudly at cache construction
(`redis_client.has_redisearch`) so its use is always visible, not silent.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

from bankassist.caching.eligibility import CacheEligibility, classify_eligibility
from bankassist.caching.stats import record as record_stat
from bankassist.caching.stats import record_latency
from bankassist.config import Settings
from bankassist.logging_config import get_logger

logger = get_logger(__name__)

_INDEX_NAME = "idx:semantic_cache"
_KEY_PREFIX = "semantic:vec:"


class QueryEmbedder(Protocol):
    def __call__(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class SemanticCacheResult:
    hit: bool
    response: str | None
    eligibility: CacheEligibility
    reason: str
    similarity: float | None = None
    source: str | None = None  # "redisearch_knn" | "python_cosine" | None


class SemanticCache:
    """Embed -> similarity search -> threshold -> serve, or fall through."""

    def __init__(
        self,
        client: Any | None,
        settings: Settings,
        embed_query: QueryEmbedder,
        *,
        redisearch_available: bool = False,
    ) -> None:
        self._client = client
        self._settings = settings
        self._embed_query = embed_query
        self._redisearch_available = redisearch_available and client is not None
        self._index_ready = False
        self.hits = 0
        self.misses = 0
        self.bypassed = 0
        self.last_latency_ms: float | None = None

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._settings.semantic_cache_enabled

    def lookup(self, query: str, *, route: str | None) -> SemanticCacheResult:
        """Look up a cached response for `query`, subject to eligibility."""
        decision = classify_eligibility(route=route, customer_scoped_tool_invoked=False)
        if decision.eligibility is not CacheEligibility.GLOBAL_CACHEABLE:
            self.bypassed += 1
            record_stat(self._client, "semantic_bypassed")
            return SemanticCacheResult(
                hit=False, response=None, eligibility=decision.eligibility, reason=decision.reason
            )

        if not self.enabled:
            self.misses += 1
            return SemanticCacheResult(
                hit=False,
                response=None,
                eligibility=decision.eligibility,
                reason="Semantic cache disabled or Redis unavailable.",
            )

        started = time.perf_counter()
        try:
            vector = self._embed_query(query)
            if self._redisearch_available:
                match = self._knn_search(vector)
            else:
                match = self._python_cosine_search(vector)
        except Exception:
            logger.warning(
                "Semantic cache lookup failed; falling through to the normal pipeline.",
                exc_info=True,
            )
            match = None
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        record_latency(self._client, self.last_latency_ms)

        if match is None:
            self.misses += 1
            record_stat(self._client, "semantic_misses")
            return SemanticCacheResult(
                hit=False,
                response=None,
                eligibility=decision.eligibility,
                reason="No cached entry met the similarity threshold.",
            )

        response, similarity, source = match
        self.hits += 1
        record_stat(self._client, "semantic_hits")
        return SemanticCacheResult(
            hit=True,
            response=response,
            eligibility=decision.eligibility,
            reason="Similarity above threshold.",
            similarity=similarity,
            source=source,
        )

    def store(
        self, query: str, response: str, *, route: str | None, customer_scoped_tool_invoked: bool
    ) -> CacheEligibility:
        """Store `response` for `query`, re-checking eligibility against what the
        request actually did (ADR-0006 rule 2: the check runs twice)."""
        decision = classify_eligibility(
            route=route, customer_scoped_tool_invoked=customer_scoped_tool_invoked
        )
        if decision.eligibility is not CacheEligibility.GLOBAL_CACHEABLE or not self.enabled:
            return decision.eligibility

        try:
            vector = self._embed_query(query)
            key = _KEY_PREFIX + hashlib.sha256(
                f"{self._settings.cache_key_version}\0{query}".encode()
            ).hexdigest()
            payload = {
                "vector": _to_bytes(vector),
                "query": query,
                "response": response,
                "route": route or "",
            }
            if self._redisearch_available:
                self._ensure_index(len(vector))
            self._client.hset(key, mapping=payload)
            self._client.expire(key, self._settings.semantic_cache_ttl_seconds)
        except Exception:
            logger.warning("Semantic cache store failed; response was not cached.", exc_info=True)

        return decision.eligibility

    # --- RediSearch path -------------------------------------------------

    def _ensure_index(self, dimensions: int) -> None:
        if self._index_ready:
            return
        try:
            self._client.ft(_INDEX_NAME).info()
            self._index_ready = True
            return
        except Exception:
            pass  # index does not exist yet; create it below

        try:
            from redis.commands.search.field import TagField, TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            schema = (
                VectorField(
                    "vector",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": dimensions, "DISTANCE_METRIC": "COSINE"},
                ),
                TextField("query"),
                TextField("response"),
                TagField("route"),
            )
            self._client.ft(_INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(prefix=[_KEY_PREFIX], index_type=IndexType.HASH),
            )
            self._index_ready = True
        except Exception:
            logger.warning(
                "Could not create the RediSearch semantic-cache index; "
                "falling back to the Python cosine scan for this process.",
                exc_info=True,
            )
            self._redisearch_available = False

    def _knn_search(self, vector: list[float]) -> tuple[str, float, str] | None:
        from redis.commands.search.query import Query

        self._ensure_index(len(vector))
        if not self._redisearch_available:
            return self._python_cosine_search(vector)

        query = (
            Query("*=>[KNN 1 @vector $vec AS score]")
            .sort_by("score")
            .return_fields("response", "score", "query")
            .dialect(2)
        )
        try:
            result = self._client.ft(_INDEX_NAME).search(
                query, query_params={"vec": _to_bytes(vector)}
            )
        except Exception:
            logger.warning("RediSearch KNN query failed; treating as a miss.", exc_info=True)
            return None

        if not result.docs:
            return None

        doc = result.docs[0]
        # RediSearch COSINE returns a *distance* (0 = identical); convert to a
        # similarity score so the same threshold semantics apply everywhere.
        distance = float(doc.score)
        similarity = 1.0 - distance
        if similarity < self._settings.semantic_cache_similarity_threshold:
            return None
        return doc.response, similarity, "redisearch_knn"

    # --- Python fallback (documented technical blocker) -------------------

    def _python_cosine_search(self, vector: list[float]) -> tuple[str, float, str] | None:
        """Bounded cosine scan used only when RediSearch is unavailable.

        Scans at most ``semantic_cache_max_candidates`` entries — acceptable at
        this project's demo scale (NFR-3), not presented as production-grade
        ANN search. See the module docstring and the Lab 7 plan's risks
        section for the explicit blocker this documents.
        """
        try:
            keys = list(self._client.scan_iter(match=f"{_KEY_PREFIX}*", count=100))
        except Exception:
            return None

        best: tuple[str, float] | None = None
        for key in keys[: self._settings.semantic_cache_max_candidates]:
            try:
                raw = self._client.hgetall(key)
            except Exception:
                continue
            if not raw:
                continue
            stored_vector = _from_bytes(raw.get(b"vector") or raw.get("vector"))
            response = raw.get(b"response") or raw.get("response")
            if stored_vector is None or response is None:
                continue
            if isinstance(response, bytes):
                response = response.decode("utf-8")

            similarity = _cosine_similarity(vector, stored_vector)
            if best is None or similarity > best[1]:
                best = (response, similarity)

        if best is None or best[1] < self._settings.semantic_cache_similarity_threshold:
            return None
        return best[0], best[1], "python_cosine"


def _to_bytes(vector: list[float]) -> bytes:
    import struct

    return struct.pack(f"{len(vector)}f", *vector)


def _from_bytes(raw: bytes | None) -> list[float] | None:
    if raw is None:
        return None
    import struct

    count = len(raw) // 4
    return list(struct.unpack(f"{count}f", raw))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
