"""``POST /api/v1/auth/login`` (Lab 4, FR-22, ADR-0010)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bankassist.api.app import create_app
from bankassist.config import Settings
from bankassist.security.passwords import hash_password
from bankassist.tools import banking_data


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "banking.db"
    with banking_data.session(db_path) as conn:
        conn.execute(
            "INSERT INTO customers (customer_id, display_name) VALUES ('CUST001', 'Asha Rao')"
        )
        conn.execute(
            "INSERT INTO users (id, username, password_hash, customer_id, role, is_active, "
            "created_at) "
            "VALUES ('USR-1', 'customer1', ?, 'CUST001', 'CUSTOMER', 1, datetime('now'))",
            (hash_password("Demo@Pass123"),),
        )
        conn.commit()

    settings = Settings(openai_api_key="test-key", banking_db_path=db_path, tracing_enabled=False)
    app = create_app(settings)
    return TestClient(app)


def test_login_with_valid_credentials_returns_token(client: TestClient):
    response = client.post(
        "/api/v1/auth/login", json={"username": "customer1", "password": "Demo@Pass123"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "CUSTOMER"
    assert body["customer_id"] == "CUST001"
    assert body["access_token"]


def test_login_with_wrong_password_is_rejected(client: TestClient):
    response = client.post(
        "/api/v1/auth/login", json={"username": "customer1", "password": "wrong"}
    )
    assert response.status_code == 401


def test_login_with_unknown_username_is_rejected_identically(client: TestClient):
    response = client.post(
        "/api/v1/auth/login", json={"username": "nobody", "password": "anything"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_error"
