from bankassist.caching.embedding_cache import EmbeddingCache


def test_miss_then_hit_on_same_model_and_text(redis_client, settings):
    cache = EmbeddingCache(redis_client, settings)

    assert cache.get("text-embedding-3-small", "hello") is None
    assert cache.misses == 1

    cache.set("text-embedding-3-small", "hello", [0.1, 0.2, 0.3])

    assert cache.get("text-embedding-3-small", "hello") == [0.1, 0.2, 0.3]
    assert cache.hits == 1


def test_key_includes_model_identifier_so_a_model_change_cannot_reuse_a_vector(
    redis_client, settings
):
    """Lab 7 amendment #2: embedding cache keys must include the model id."""
    cache = EmbeddingCache(redis_client, settings)
    cache.set("text-embedding-3-small", "hello", [0.1, 0.2])

    # Same text, different model: must miss, not silently return the other
    # model's (differently-dimensioned, differently-meaning) vector.
    assert cache.get("text-embedding-3-large", "hello") is None


def test_disabled_cache_is_always_a_miss_and_never_writes(redis_client, settings):
    settings = settings.model_copy(update={"embedding_cache_enabled": False})
    cache = EmbeddingCache(redis_client, settings)

    cache.set("text-embedding-3-small", "hello", [0.1])
    assert cache.get("text-embedding-3-small", "hello") is None


def test_no_client_degrades_to_a_miss_without_raising(settings):
    cache = EmbeddingCache(None, settings)
    assert cache.get("text-embedding-3-small", "hello") is None
    cache.set("text-embedding-3-small", "hello", [0.1])  # must not raise


def test_redis_failure_is_treated_as_a_miss_not_an_exception(settings):
    class ExplodingClient:
        def get(self, key):
            raise ConnectionError("redis is down")

        def set(self, *args, **kwargs):
            raise ConnectionError("redis is down")

    cache = EmbeddingCache(ExplodingClient(), settings)
    assert cache.get("text-embedding-3-small", "hello") is None
    cache.set("text-embedding-3-small", "hello", [0.1])  # must not raise
