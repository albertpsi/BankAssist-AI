"""`GET /api/v1/cache/stats` through a real ASGI client (Lab 7, ADR-0013)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bankassist.api.app import create_app
from bankassist.caching.stats import record
from bankassist.config import Settings


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_cache_stats_with_redis_disabled_returns_all_zeros(client: TestClient) -> None:
    """REDIS_ENABLED is false by default (CLAUDE.md §4's "off unless explicitly
    configured" posture) — the endpoint must still respond, not 500."""
    response = client.get("/api/v1/cache/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["semantic_hits"] == 0
    assert body["average_redis_latency_ms"] == 0.0


def test_cache_stats_reflects_recorded_counters(app: FastAPI, client: TestClient) -> None:
    import fakeredis

    fake_client = fakeredis.FakeStrictRedis()
    app.state.redis_client = fake_client
    record(fake_client, "semantic_hits")
    record(fake_client, "semantic_hits")
    record(fake_client, "embedding_misses")

    response = client.get("/api/v1/cache/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["semantic_hits"] == 2
    assert body["embedding_misses"] == 1
    assert body["estimated_openai_calls_saved"] == 2
