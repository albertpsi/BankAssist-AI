"""``POST /api/v1/agent/chat`` and ``/resume`` (Lab 4, FR-19/FR-20)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bankassist.agents.graph import build_graph
from bankassist.api.app import create_app
from bankassist.config import Settings
from bankassist.llm.stub import StubLLMClient
from bankassist.security.jwt_tokens import issue_token
from bankassist.security.passwords import hash_password
from bankassist.tools import banking_data

DISPUTE_ROUTE = '{"route": "DISPUTE", "confidence": 0.9, "reason": "unrecognized txn"}'
BANKING_ROUTE = '{"route": "BANKING", "confidence": 0.9, "reason": "own transactions"}'


@dataclass
class _FakeResult:
    generated_answer: str = "KYC requires a PAN card."
    citations: list[str] = field(default_factory=lambda: ["kyc.md"])


class _FakePipeline:
    def answer(self, question: str) -> _FakeResult:
        return _FakeResult()


def _seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "banking.db"
    with banking_data.session(db_path) as conn:
        conn.executemany(
            "INSERT INTO customers (customer_id, display_name) VALUES (?, ?)",
            [("CUST001", "Asha Rao"), ("CUST002", "Vikram Nair")],
        )
        conn.execute(
            "INSERT INTO accounts (account_id, customer_id, account_type, balance_paise, currency) "
            "VALUES ('ACC-1', 'CUST001', 'SAVINGS', 100000, 'INR')"
        )
        conn.execute(
            "INSERT INTO transactions (transaction_id, customer_id, account_id, card_id, "
            "amount_paise, currency, merchant, category, txn_date, status) VALUES "
            "('TX1007', 'CUST001', 'ACC-1', NULL, 450000, 'INR', 'QuickPay', 'Shopping', "
            "'2026-08-01', 'POSTED')"
        )
        conn.execute(
            "INSERT INTO users (id, username, password_hash, customer_id, role, is_active, "
            "created_at) "
            "VALUES ('USR-1', 'customer1', ?, 'CUST001', 'CUSTOMER', 1, datetime('now'))",
            (hash_password("Demo@Pass123"),),
        )
        conn.commit()
    return db_path


def _app_with_graph(tmp_path: Path, responses: list[str]) -> tuple[FastAPI, Settings]:
    db_path = _seeded_db(tmp_path)
    settings = Settings(openai_api_key="test-key", banking_db_path=db_path, tracing_enabled=False)
    app = create_app(settings)
    llm = StubLLMClient(responses)
    app.state.agent_graph = build_graph(
        llm=llm, enterprise_pipeline=_FakePipeline(), db_path=db_path
    )
    return app, settings


def _token(settings: Settings, *, role: str, customer_id: str | None) -> str:
    return issue_token(settings=settings, user_id="USR-1", role=role, customer_id=customer_id)


def test_chat_without_token_is_rejected(tmp_path: Path):
    app, _ = _app_with_graph(tmp_path, [BANKING_ROUTE])
    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "show my transactions", "session_id": "S1"},
    )
    assert response.status_code == 401


def test_banking_chat_returns_scoped_events(tmp_path: Path):
    app, settings = _app_with_graph(tmp_path, [BANKING_ROUTE])
    client = TestClient(app)
    token = _token(settings, role="CUSTOMER", customer_id="CUST001")

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "show my transactions", "session_id": "S1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "banking_agent"
    assert body["status"] == "completed"
    labels = [e["label"] for e in body["execution_events"]]
    assert "get_recent_transactions" in labels


def test_chat_body_customer_id_mismatch_is_rejected(tmp_path: Path):
    app, settings = _app_with_graph(tmp_path, [BANKING_ROUTE])
    client = TestClient(app)
    token = _token(settings, role="CUSTOMER", customer_id="CUST001")

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "show my transactions", "customer_id": "CUST002", "session_id": "S1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_dispute_flow_waits_then_resumes_and_creates_dispute(tmp_path: Path):
    app, settings = _app_with_graph(tmp_path, [DISPUTE_ROUTE, DISPUTE_ROUTE])
    client = TestClient(app)
    token = _token(settings, role="CUSTOMER", customer_id="CUST001")
    headers = {"Authorization": f"Bearer {token}"}

    turn1 = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "I don't recognize one of my recent transactions.",
            "session_id": "S-DEMO",
        },
        headers=headers,
    )
    assert turn1.status_code == 200
    turn1_body = turn1.json()
    assert turn1_body["status"] == "completed"
    offered = turn1_body["available_transactions"]
    assert offered is not None
    assert any(t["transaction_id"] == "TX1007" for t in offered)

    turn2 = client.post(
        "/api/v1/agent/chat",
        json={"message": "The ₹4,500 one.", "session_id": "S-DEMO"},
        headers=headers,
    )
    assert turn2.status_code == 200
    body = turn2.json()
    assert body["status"] == "waiting_approval"
    assert body["approval_required"] is True
    assert body["available_transactions"] is None

    approve = client.post(
        "/api/v1/agent/resume",
        json={"session_id": "S-DEMO", "approved": True},
        headers=headers,
    )
    assert approve.status_code == 200
    approve_body = approve.json()
    assert approve_body["status"] == "completed"
    assert "Dispute created" in approve_body["answer"]


def test_resuming_an_already_resolved_interrupt_is_rejected(tmp_path: Path):
    app, settings = _app_with_graph(tmp_path, [DISPUTE_ROUTE, DISPUTE_ROUTE])
    client = TestClient(app)
    token = _token(settings, role="CUSTOMER", customer_id="CUST001")
    headers = {"Authorization": f"Bearer {token}"}

    client.post(
        "/api/v1/agent/chat",
        json={"message": "I don't recognize one of my recent transactions.", "session_id": "S-DUP"},
        headers=headers,
    )
    client.post(
        "/api/v1/agent/chat",
        json={"message": "The ₹4,500 one.", "session_id": "S-DUP"},
        headers=headers,
    )
    first = client.post(
        "/api/v1/agent/resume", json={"session_id": "S-DUP", "approved": True}, headers=headers
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/agent/resume", json={"session_id": "S-DUP", "approved": True}, headers=headers
    )
    assert second.status_code == 409
