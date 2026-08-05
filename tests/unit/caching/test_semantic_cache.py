from bankassist.caching.eligibility import CacheEligibility
from bankassist.caching.semantic_cache import SemanticCache


def _embed(text: str) -> list[float]:
    """A tiny deterministic fake embedder: near-identical text -> near-identical
    vector, dissimilar text -> an orthogonal vector. Good enough to exercise
    threshold behaviour without a real model."""
    if "dispute window" in text:
        return [1.0, 0.0, 0.0]
    if "chargeback period" in text:
        return [0.99, 0.01, 0.0]  # semantically similar rephrasing
    return [0.0, 1.0, 0.0]  # unrelated


def test_miss_on_an_empty_cache(redis_client, settings):
    cache = SemanticCache(redis_client, settings, _embed)

    result = cache.lookup("what is the dispute window?", route="POLICY")

    assert result.hit is False
    assert result.eligibility is CacheEligibility.GLOBAL_CACHEABLE


def test_store_then_hit_on_a_near_identical_query(redis_client, settings):
    cache = SemanticCache(redis_client, settings, _embed)
    cache.store(
        "what is the dispute window?",
        "You have 30 days to dispute a transaction.",
        route="POLICY",
        customer_scoped_tool_invoked=False,
    )

    result = cache.lookup("what is the chargeback period?", route="POLICY")

    assert result.hit is True
    assert result.response == "You have 30 days to dispute a transaction."
    assert result.similarity >= settings.semantic_cache_similarity_threshold


def test_dissimilar_query_is_a_miss_even_with_a_populated_cache(redis_client, settings):
    cache = SemanticCache(redis_client, settings, _embed)
    cache.store(
        "what is the dispute window?",
        "You have 30 days to dispute a transaction.",
        route="POLICY",
        customer_scoped_tool_invoked=False,
    )

    result = cache.lookup("something completely unrelated", route="POLICY")

    assert result.hit is False


def test_threshold_boundary(redis_client, settings):
    settings = settings.model_copy(update={"semantic_cache_similarity_threshold": 0.9999999})
    cache = SemanticCache(redis_client, settings, _embed)
    cache.store(
        "what is the dispute window?",
        "You have 30 days to dispute a transaction.",
        route="POLICY",
        customer_scoped_tool_invoked=False,
    )

    # The rephrasing's cosine similarity to [1,0,0] is ~0.99995 — below an
    # unreasonably strict threshold, so this must miss.
    result = cache.lookup("what is the chargeback period?", route="POLICY")
    assert result.hit is False


def test_customer_scoped_route_is_bypassed_on_lookup(redis_client, settings):
    cache = SemanticCache(redis_client, settings, _embed)

    result = cache.lookup("what is my balance?", route="BANKING")

    assert result.hit is False
    assert result.eligibility is CacheEligibility.NOT_CACHEABLE


def test_a_policy_looking_request_that_invoked_a_tool_is_never_stored(redis_client, settings):
    """ADR-0006 rule 2, re-checked before store: a request classified as POLICY
    that ended up invoking a customer-scoped tool must not be written to the
    cache, and must never be served to a later, different customer."""
    cache = SemanticCache(redis_client, settings, _embed)

    eligibility = cache.store(
        "what is the dispute window?",
        "Customer A's balance is 10,000.",
        route="POLICY",
        customer_scoped_tool_invoked=True,
    )

    assert eligibility is CacheEligibility.NOT_CACHEABLE
    result = cache.lookup("what is the dispute window?", route="POLICY")
    assert result.hit is False


def test_output_of_a_cache_hit_is_never_trusted_blindly_by_this_module(redis_client, settings):
    """ADR-0006 rule 4 ('output guardrails run on cache hits') is enforced by
    the caller (agents/graph.py's output_guardrails node), not here — this
    test only documents that `SemanticCache` returns the raw stored text and
    performs no guardrail pass of its own, so callers must not skip theirs."""
    cache = SemanticCache(redis_client, settings, _embed)
    cache.store(
        "what is the dispute window?",
        "raw unguarded text",
        route="POLICY",
        customer_scoped_tool_invoked=False,
    )
    result = cache.lookup("what is the dispute window?", route="POLICY")
    assert result.response == "raw unguarded text"


def test_disabled_cache_is_always_a_miss(redis_client, settings):
    settings = settings.model_copy(update={"semantic_cache_enabled": False})
    cache = SemanticCache(redis_client, settings, _embed)
    cache.store(
        "what is the dispute window?", "answer", route="POLICY", customer_scoped_tool_invoked=False
    )
    result = cache.lookup("what is the dispute window?", route="POLICY")
    assert result.hit is False


def test_redis_unavailable_degrades_to_a_miss_without_raising(settings):
    cache = SemanticCache(None, settings, _embed)
    cache.store(
        "what is the dispute window?", "answer", route="POLICY", customer_scoped_tool_invoked=False
    )  # must not raise
    result = cache.lookup("what is the dispute window?", route="POLICY")
    assert result.hit is False


def test_redisearch_index_creation_failure_falls_back_to_python_cosine(redis_client, settings):
    """Simulates the documented technical blocker: RediSearch advertised as
    available but index creation fails (e.g. an old/partial module build)."""
    cache = SemanticCache(redis_client, settings, _embed, redisearch_available=True)

    # fakeredis has no FT.* commands at all, so `_ensure_index` must catch the
    # failure and fall back rather than propagate it.
    cache.store(
        "what is the dispute window?",
        "You have 30 days to dispute a transaction.",
        route="POLICY",
        customer_scoped_tool_invoked=False,
    )
    result = cache.lookup("what is the chargeback period?", route="POLICY")

    assert result.hit is True
    assert result.source == "python_cosine"
