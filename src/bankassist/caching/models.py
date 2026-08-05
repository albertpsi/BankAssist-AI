"""Shared pydantic models for the Lab 7 caching layer (ADR-0013)."""

from __future__ import annotations

from pydantic import BaseModel


class CacheStats(BaseModel):
    """Aggregate counters served by ``GET /api/v1/cache/stats``."""

    semantic_hits: int = 0
    semantic_misses: int = 0
    semantic_bypassed: int = 0
    embedding_hits: int = 0
    embedding_misses: int = 0
    tool_hits: int = 0
    tool_misses: int = 0
    tool_bypassed: int = 0
    estimated_openai_calls_saved: int = 0
    estimated_embedding_calls_saved: int = 0
    average_redis_latency_ms: float = 0.0
