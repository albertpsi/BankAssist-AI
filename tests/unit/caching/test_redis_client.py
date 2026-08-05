from bankassist.caching.redis_client import build_redis_client, has_redisearch
from bankassist.config import Settings


def test_disabled_by_default_returns_none():
    settings = Settings(openai_api_key="sk-test-not-used", _env_file=None)
    assert settings.redis_enabled is False
    assert build_redis_client(settings) is None


def test_enabled_but_unreachable_returns_none_not_an_exception():
    settings = Settings(
        openai_api_key="sk-test-not-used",
        redis_enabled=True,
        redis_url="redis://127.0.0.1:1/0",  # a port nothing listens on
        redis_connect_timeout_seconds=0.2,
        redis_socket_timeout_seconds=0.2,
        _env_file=None,
    )
    assert build_redis_client(settings) is None


def test_has_redisearch_is_false_for_a_client_missing_the_module(redis_client):
    # fakeredis does not implement MODULE LIST at all — the exact case the
    # amendment #1 fallback path is documented for.
    assert has_redisearch(redis_client) is False


def test_has_redisearch_is_false_for_no_client():
    assert has_redisearch(None) is False
