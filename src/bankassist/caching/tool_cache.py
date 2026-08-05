"""Tool response cache (Lab 7, ADR-0013).

Only deterministic, non-customer-scoped tools are ever eligible — see the
cacheability matrix in the Lab 7 plan. Every scoped tool in
``tools/scoped_tools.py`` takes a ``SecurityContext`` and is therefore excluded
by construction: it is never passed to this cache because the dispatcher only
consults ``TOOL_CACHE_ALLOWLIST`` (a positive allowlist, not a blocklist — same
fail-closed default as ADR-0006) and none of those labels appear in it.
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

# Positive allowlist: a tool must be named here to ever be cached. Adding a new
# tool defaults it to *not* cached until someone deliberately lists it — the
# failure mode of forgetting is a cache miss, never a stale/incorrect answer
# for a tool that turns out to be customer-scoped or mutable.
TOOL_CACHE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "policy_lookup",
        "faq_lookup",
        "card_agreement_lookup",
        "kyc_document_lookup",
        "rbi_rules_lookup",
    }
)


class ToolCache:
    """Versioned, argument-hash-keyed cache for deterministic tool calls.

    Keys are built from the tool's label, the configured cache version, and a
    hash of its arguments — never an entity id (Lab 7 amendment #3). Two calls
    to the same tool with the same arguments hash identically regardless of
    what any particular argument happens to be named or shaped like an id.
    """

    def __init__(self, client: Any | None, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.hits = 0
        self.misses = 0
        self.bypassed = 0
        self.last_latency_ms: float | None = None

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._settings.tool_cache_enabled

    def is_cacheable(self, label: str) -> bool:
        return label in TOOL_CACHE_ALLOWLIST

    def get(self, label: str, args: dict[str, Any]) -> Any | None:
        if not self.enabled or not self.is_cacheable(label):
            self.bypassed += 1
            record_stat(self._client, "tool_bypassed")
            return None

        key = self._key(label, args)
        started = time.perf_counter()
        try:
            raw = self._client.get(key)
        except Exception:
            logger.warning(
                "Tool cache GET failed for %s; treating as a miss.", label, exc_info=True
            )
            raw = None
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        record_latency(self._client, self.last_latency_ms)

        if raw is None:
            self.misses += 1
            record_stat(self._client, "tool_misses")
            return None

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Tool cache entry for %s was unparseable; treating as a miss.", label)
            self.misses += 1
            record_stat(self._client, "tool_misses")
            return None

        self.hits += 1
        record_stat(self._client, "tool_hits")
        return payload

    def set(self, label: str, args: dict[str, Any], result: Any) -> None:
        if not self.enabled or not self.is_cacheable(label):
            return

        key = self._key(label, args)
        try:
            self._client.set(key, json.dumps(result), ex=self._settings.tool_cache_ttl_seconds)
        except (TypeError, ValueError):
            # Not everything a tool returns is JSON-serializable as-is; callers
            # are expected to pass a plain-dict projection. Skip caching rather
            # than fail the request over it.
            logger.warning("Tool result for %s was not JSON-serializable; not cached.", label)
        except Exception:
            logger.warning(
                "Tool cache SET failed for %s; continuing without caching.", label, exc_info=True
            )

    def _key(self, label: str, args: dict[str, Any]) -> str:
        canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"tool:{label}:{self._settings.cache_key_version}:{digest}"
