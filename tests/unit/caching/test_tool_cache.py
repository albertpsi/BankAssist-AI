from bankassist.caching.tool_cache import TOOL_CACHE_ALLOWLIST, ToolCache


def test_allowlisted_tool_misses_then_hits(redis_client, settings):
    cache = ToolCache(redis_client, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))

    assert cache.get(label, {"doc_id": "faq-1"}) is None
    assert cache.misses == 1

    cache.set(label, {"doc_id": "faq-1"}, {"text": "answer"})

    assert cache.get(label, {"doc_id": "faq-1"}) == {"text": "answer"}
    assert cache.hits == 1


def test_key_is_argument_hash_based_not_entity_id(redis_client, settings):
    """Lab 7 amendment #3: keys are versioned argument-hashes, never a raw id."""
    cache = ToolCache(redis_client, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))
    key = cache._key(label, {"doc_id": "faq-1"})  # noqa: SLF001

    assert "faq-1" not in key  # the literal id never appears in the key
    assert key.startswith(f"tool:{label}:{settings.cache_key_version}:")


def test_argument_order_does_not_change_the_key(redis_client, settings):
    cache = ToolCache(redis_client, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))

    key_a = cache._key(label, {"a": 1, "b": 2})  # noqa: SLF001
    key_b = cache._key(label, {"b": 2, "a": 1})  # noqa: SLF001

    assert key_a == key_b


def test_a_scoped_customer_tool_label_is_never_cached_even_if_offered(redis_client, settings):
    """Fail-closed regression: a caller mistakenly passing a scoped tool's
    label must never be served from, or written to, the cache."""
    cache = ToolCache(redis_client, settings)

    assert cache.is_cacheable("get_recent_transactions") is False
    assert cache.is_cacheable("create_dispute") is False

    cache.set("get_recent_transactions", {"customer_id": "cust-1"}, {"leaked": "data"})
    assert cache.get("get_recent_transactions", {"customer_id": "cust-1"}) is None
    assert cache.bypassed == 1  # `set()` bypasses silently; `get()` counts the bypass


def test_version_bump_invalidates_prior_entries(redis_client, settings):
    cache_v1 = ToolCache(redis_client, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))
    cache_v1.set(label, {"doc_id": "faq-1"}, {"text": "old"})

    settings_v2 = settings.model_copy(update={"cache_key_version": "v2"})
    cache_v2 = ToolCache(redis_client, settings_v2)

    assert cache_v2.get(label, {"doc_id": "faq-1"}) is None


def test_ttl_is_set_on_store(redis_client, settings):
    cache = ToolCache(redis_client, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))
    cache.set(label, {"doc_id": "faq-1"}, {"text": "answer"})

    ttl = redis_client.ttl(cache._key(label, {"doc_id": "faq-1"}))  # noqa: SLF001
    assert 0 < ttl <= settings.tool_cache_ttl_seconds


def test_disabled_cache_never_hits(redis_client, settings):
    settings = settings.model_copy(update={"tool_cache_enabled": False})
    cache = ToolCache(redis_client, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))

    cache.set(label, {"doc_id": "faq-1"}, {"text": "answer"})
    assert cache.get(label, {"doc_id": "faq-1"}) is None


def test_redis_unavailable_falls_back_to_a_miss(settings):
    cache = ToolCache(None, settings)
    label = next(iter(TOOL_CACHE_ALLOWLIST))

    cache.set(label, {"doc_id": "faq-1"}, {"text": "answer"})  # must not raise
    assert cache.get(label, {"doc_id": "faq-1"}) is None
