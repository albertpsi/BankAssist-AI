"""Cache statistics (Lab 7, ADR-0013) — backs `GET /api/v1/cache/stats`.

Counters live in Redis (`cache:stats:*`), not in process memory, because the
FastAPI process and the Streamlit process are separate (CLAUDE.md §4) and both
need to see the same numbers. Each cache module calls `record()` at its own
hit/miss/bypass points; this module only aggregates and estimates savings.
"""

from __future__ import annotations

from typing import Any

from bankassist.caching.models import CacheStats
from bankassist.config import Settings
from bankassist.logging_config import get_logger

logger = get_logger(__name__)

_COUNTER_NAMES = (
    "semantic_hits",
    "semantic_misses",
    "semantic_bypassed",
    "embedding_hits",
    "embedding_misses",
    "tool_hits",
    "tool_misses",
    "tool_bypassed",
)


def record(client: Any | None, counter: str) -> None:
    """Increment one named counter. A no-op, never a failure, without Redis."""
    if client is None:
        return
    try:
        client.incr(f"cache:stats:{counter}")
    except Exception:
        logger.debug("Cache stats counter increment failed for %s", counter, exc_info=True)


def record_latency(client: Any | None, latency_ms: float) -> None:
    """Fold one Redis round-trip's latency into a running average.

    Stored as a (sum, count) pair rather than a single average so the average
    stays correct across restarts/multiple processes.
    """
    if client is None:
        return
    try:
        pipe = client.pipeline()
        pipe.incrbyfloat("cache:stats:redis_latency_ms_sum", latency_ms)
        pipe.incr("cache:stats:redis_latency_ms_count")
        pipe.execute()
    except Exception:
        logger.debug("Cache stats latency recording failed", exc_info=True)


def get_stats(client: Any | None, settings: Settings) -> CacheStats:  # noqa: ARG001
    """Read the aggregate counters, estimating savings from configured assumptions.

    ``settings`` is accepted for a stable call signature across every caller
    (routes, Streamlit, tests) even though this function does not currently
    read any field off it — the savings estimate is a direct hit count, not
    derived from configured assumptions.
    """
    if client is None:
        return CacheStats()

    try:
        raw = {name: client.get(f"cache:stats:{name}") for name in _COUNTER_NAMES}
        latency_sum = client.get("cache:stats:redis_latency_ms_sum")
        latency_count = client.get("cache:stats:redis_latency_ms_count")
    except Exception:
        logger.warning("Could not read cache stats from Redis.", exc_info=True)
        return CacheStats()

    values = {name: int(value or 0) for name, value in raw.items()}
    count = int(latency_count or 0)
    average_latency = (float(latency_sum or 0.0) / count) if count else 0.0

    # A semantic-cache hit skips one full LLM generation call; an embedding
    # cache hit skips one embeddings API call — both are direct counts, not
    # extrapolations, unlike the demo assumptions in the Lab 7 plan doc.
    return CacheStats(
        semantic_hits=values["semantic_hits"],
        semantic_misses=values["semantic_misses"],
        semantic_bypassed=values["semantic_bypassed"],
        embedding_hits=values["embedding_hits"],
        embedding_misses=values["embedding_misses"],
        tool_hits=values["tool_hits"],
        tool_misses=values["tool_misses"],
        tool_bypassed=values["tool_bypassed"],
        estimated_openai_calls_saved=values["semantic_hits"],
        estimated_embedding_calls_saved=values["embedding_hits"],
        average_redis_latency_ms=round(average_latency, 3),
    )
