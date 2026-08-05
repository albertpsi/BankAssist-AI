"""Lab 7 cost-optimization caching layer (ADR-0013).

Three independent Redis-backed caches — semantic response cache, embedding cache,
tool response cache — plus the shared eligibility model they all consult. Every
cache degrades to a no-op (never an exception) when Redis is disabled or
unreachable, mirroring the existing "off unless explicitly configured, and never
takes down the app" posture used for Pinecone (``rag/vector_store.py``) and
AgentOps (``observability/agentops_client.py``).

Nothing in this package is imported by name into ``rag``, ``agents``, or
``tools`` at *class definition* time in a way that would make those packages
require Redis to import — each collaborator is optional and constructor-injected
(``embedding_cache: EmbeddingCache | None = None``), so the existing test suite
and any Redis-less environment keeps working unmodified.
"""

from __future__ import annotations

from bankassist.caching.eligibility import CacheEligibility, classify_eligibility
from bankassist.caching.embedding_cache import EmbeddingCache
from bankassist.caching.redis_client import build_redis_client
from bankassist.caching.semantic_cache import SemanticCache, SemanticCacheResult
from bankassist.caching.stats import CacheStats
from bankassist.caching.tool_cache import TOOL_CACHE_ALLOWLIST, ToolCache

__all__ = [
    "TOOL_CACHE_ALLOWLIST",
    "CacheEligibility",
    "CacheStats",
    "EmbeddingCache",
    "SemanticCache",
    "SemanticCacheResult",
    "ToolCache",
    "build_redis_client",
    "classify_eligibility",
]
