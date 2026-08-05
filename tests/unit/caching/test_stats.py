from bankassist.caching.stats import get_stats, record, record_latency


def test_stats_start_at_zero(redis_client, settings):
    stats = get_stats(redis_client, settings)
    assert stats.semantic_hits == 0
    assert stats.average_redis_latency_ms == 0.0


def test_record_increments_the_right_counter(redis_client, settings):
    record(redis_client, "semantic_hits")
    record(redis_client, "semantic_hits")
    record(redis_client, "semantic_misses")

    stats = get_stats(redis_client, settings)
    assert stats.semantic_hits == 2
    assert stats.semantic_misses == 1
    assert stats.estimated_openai_calls_saved == 2


def test_record_latency_computes_a_running_average(redis_client, settings):
    record_latency(redis_client, 10.0)
    record_latency(redis_client, 30.0)

    stats = get_stats(redis_client, settings)
    assert stats.average_redis_latency_ms == 20.0


def test_embedding_cache_records_latency_into_shared_stats(redis_client, settings):
    """Regression: each cache measures its own `last_latency_ms` but must also
    push it into the shared Redis average — this was silently disconnected."""
    from bankassist.caching.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(redis_client, settings)
    cache.get("text-embedding-3-small", "hello")  # a miss still round-trips Redis

    stats = get_stats(redis_client, settings)
    assert stats.average_redis_latency_ms >= 0.0
    assert redis_client.get("cache:stats:redis_latency_ms_count") is not None


def test_tool_cache_records_latency_into_shared_stats(redis_client, settings):
    from bankassist.caching.tool_cache import TOOL_CACHE_ALLOWLIST, ToolCache

    cache = ToolCache(redis_client, settings)
    cache.get(next(iter(TOOL_CACHE_ALLOWLIST)), {"doc_id": "faq-1"})

    assert redis_client.get("cache:stats:redis_latency_ms_count") is not None


def test_semantic_cache_records_latency_into_shared_stats(redis_client, settings):
    from bankassist.caching.semantic_cache import SemanticCache

    cache = SemanticCache(redis_client, settings, lambda text: [0.0, 1.0, 0.0])  # noqa: ARG005
    cache.lookup("what is the dispute window?", route="POLICY")

    assert redis_client.get("cache:stats:redis_latency_ms_count") is not None


def test_no_client_returns_zeroed_stats_without_raising(settings):
    stats = get_stats(None, settings)
    assert stats.semantic_hits == 0


def test_record_without_client_does_not_raise():
    record(None, "semantic_hits")  # must not raise
    record_latency(None, 5.0)  # must not raise
