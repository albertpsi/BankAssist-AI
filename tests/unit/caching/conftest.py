import fakeredis
import pytest

from bankassist.config import Settings


@pytest.fixture
def redis_client():
    """A real (in-memory) Redis protocol implementation, not a mock.

    `fakeredis` does not implement `MODULE LIST` (RediSearch detection) or the
    `FT.*` search commands — this stands in for the documented fallback path
    (plain Redis, no RediSearch module), which is exactly the scenario the Lab
    7 amendment requires being explicit about.
    """
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def settings():
    return Settings(
        openai_api_key="sk-test-not-used",
        redis_enabled=True,
        _env_file=None,
    )
