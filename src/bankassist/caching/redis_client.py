"""Redis connection factory (Lab 7, ADR-0013).

The only module in the application that imports ``redis`` — same "one module
owns the SDK" convention as ``rag/vector_store.py`` (Pinecone) and
``rag/embeddings.py`` (OpenAI). Every cache module talks to Redis only through
the client this module builds, never by importing ``redis`` directly.
"""

from __future__ import annotations

from typing import Any

from bankassist.config import Settings
from bankassist.logging_config import get_logger

logger = get_logger(__name__)


def build_redis_client(settings: Settings) -> Any | None:
    """Build a Redis client from configuration, or ``None`` if caching is off.

    Returns ``None`` rather than raising when ``redis_enabled`` is ``False`` —
    the same "off unless explicitly configured" posture as
    ``PineconeVectorStore``/AgentOps. A connection failure at construction time
    is also swallowed and logged: every cache built on top of this must treat a
    ``None``/unreachable client as a cache miss, never as an error the request
    fails on (§9 of the Lab 7 plan, "Redis unavailable").
    """
    if not settings.redis_enabled:
        return None

    try:
        import redis
    except ImportError:
        logger.warning(
            "REDIS_ENABLED is true but the `redis` package is not installed; "
            "caching stays disabled."
        )
        return None

    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=settings.redis_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            decode_responses=False,
        )
        client.ping()
    except Exception:
        logger.warning(
            "Could not connect to Redis at the configured REDIS_URL; caching "
            "stays disabled for this process.",
            exc_info=True,
        )
        return None

    logger.info("Redis cache client connected", extra={"redis_url": _safe_url(settings.redis_url)})
    return client


def has_redisearch(client: Any) -> bool:
    """True when the connected Redis server exposes the RediSearch module.

    Checked once at startup so ``SemanticCache`` can pick vector KNN search
    (the required approach per the Lab 7 amendment) over the documented Python
    fallback, rather than probing on every request.
    """
    if client is None:
        return False
    try:
        modules = client.execute_command("MODULE LIST")
    except Exception:
        logger.warning(
            "Could not query Redis for loaded modules (MODULE LIST failed); "
            "assuming RediSearch is unavailable and falling back to the "
            "documented Python-side similarity scan. This is expected against "
            "plain `redis:*` images — use `redis/redis-stack-server` to enable "
            "native vector search.",
            exc_info=True,
        )
        return False

    names = set()
    for entry in modules or []:
        # RESP2 gives a flat list of alternating key/value pairs per module;
        # RESP3/redis-py may give dicts — handle both defensively.
        if isinstance(entry, dict):
            name = entry.get(b"name") or entry.get("name")
        else:
            entry = list(entry)
            name = None
            for index in range(0, len(entry) - 1, 2):
                if entry[index] in (b"name", "name"):
                    name = entry[index + 1]
                    break
        if name:
            names.add(name.decode() if isinstance(name, bytes) else name)

    found = "search" in names
    if not found:
        logger.warning(
            "Redis is reachable but the `search` (RediSearch) module is not "
            "loaded — documented technical blocker (Lab 7 amendment #1). "
            "Falling back to a bounded Python-side cosine similarity scan for "
            "the semantic cache. Run `redis/redis-stack-server` locally to "
            "enable native vector KNN search instead.",
        )
    return found


def _safe_url(url: str) -> str:
    """Strip credentials from a Redis URL before it ever reaches a log line."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host_part = rest.partition("@")
    return f"{scheme}://***@{host_part}"
