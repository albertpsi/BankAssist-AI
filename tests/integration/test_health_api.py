"""The HTTP surface, through a real ASGI client (FR-L1-5)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bankassist.api.app import TRACE_HEADER, create_app
from bankassist.config import Settings
from bankassist.errors import LLMError
from bankassist.tracing.span import SpanStatus


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_health_returns_200_with_the_documented_body(
    client: TestClient, settings: Settings
) -> None:
    """AC-L1-5."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": "test",
        "llm_provider": "openai",
    }


def test_health_never_exposes_the_credential(client: TestClient) -> None:
    """AC-L1-4 at the HTTP boundary."""
    body = client.get("/health").text

    assert "sk-test-not-a-real-key" not in body
    assert "api_key" not in body.lower()


def test_health_does_not_require_a_working_provider(settings: Settings) -> None:
    """It must stay usable as a liveness probe when the provider is unreachable."""
    response = TestClient(create_app(settings)).get("/health")

    assert response.status_code == 200


def test_response_carries_a_trace_id_header(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers[TRACE_HEADER]


def test_each_request_gets_a_distinct_trace_id(client: TestClient) -> None:
    first = client.get("/health").headers[TRACE_HEADER]
    second = client.get("/health").headers[TRACE_HEADER]

    assert first != second


def test_inbound_trace_id_is_honoured(client: TestClient) -> None:
    """Lets a caller correlate across a whole conversation."""
    response = client.get("/health", headers={TRACE_HEADER: "caller-supplied-id"})

    assert response.headers[TRACE_HEADER] == "caller-supplied-id"


def test_request_emits_a_span(app: FastAPI, client: TestClient) -> None:
    client.get("/health")

    spans = app.state.tracer.spans()
    assert [span.name for span in spans] == ["GET /health"]
    assert spans[0].attributes["status_code"] == 200


def test_unknown_route_uses_the_error_envelope(client: TestClient) -> None:
    """AC-L1-6: a 404 must not be FastAPI's default shape."""
    response = client.get("/no-such-route")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "http_error"
    assert body["trace_id"]


def test_application_error_maps_to_its_declared_status(app: FastAPI) -> None:
    """AC-L1-7."""

    @app.get("/boom-known")
    def _boom() -> None:
        raise LLMError("provider is down", details={"provider": "openai"})

    response = TestClient(app).get("/boom-known")

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "llm_error"
    assert body["error"]["message"] == "provider is down"
    assert body["error"]["details"] == {"provider": "openai"}


def test_unexpected_error_is_sanitized(app: FastAPI) -> None:
    """AC-L1-8: internal detail is logged, never returned."""

    @app.get("/boom-unexpected")
    def _boom() -> None:
        raise RuntimeError("connection string postgres://user:hunter2@internal-db")

    response = TestClient(app, raise_server_exceptions=False).get("/boom-unexpected")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "An internal error occurred."
    assert "hunter2" not in response.text
    assert "RuntimeError" not in response.text


def test_unexpected_error_is_still_correlatable(app: FastAPI) -> None:
    """A 500 is where a trace id matters most; it must not be lost on that path."""

    @app.get("/boom-correlated")
    def _boom() -> None:
        raise RuntimeError("kaboom")

    response = TestClient(app, raise_server_exceptions=False).get("/boom-correlated")

    assert response.status_code == 500
    assert response.headers[TRACE_HEADER]
    assert response.json()["trace_id"] == response.headers[TRACE_HEADER]


def test_failed_request_span_is_recorded_as_an_error(app: FastAPI) -> None:
    @app.get("/boom-span")
    def _boom() -> None:
        raise RuntimeError("kaboom")

    TestClient(app, raise_server_exceptions=False).get("/boom-span")

    (span,) = app.state.tracer.spans()
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "RuntimeError"


def test_validation_error_uses_the_error_envelope(app: FastAPI) -> None:
    """Every failure path returns one shape, including request validation."""

    @app.get("/needs-a-number")
    def _needs_a_number(count: int) -> dict[str, int]:
        return {"count": count}

    response = TestClient(app).get("/needs-a-number", params={"count": "not-a-number"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["errors"][0]["loc"] == ["query", "count"]
    assert body["trace_id"]


def test_openapi_docs_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_openapi_schema_documents_health(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/health" in schema["paths"]
