from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from bankassist.security.context import SecurityContext
from bankassist.tools import banking_data


@dataclass
class FakePipelineResult:
    generated_answer: str = "KYC requires a PAN card and proof of address."
    citations: list[str] = field(default_factory=lambda: ["kyc-policy.md"])


class FakeEnterpriseRagPipeline:
    """Duck-types ``EnterpriseRagPipeline.answer`` without any retrieval/LLM call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer(self, question: str) -> FakePipelineResult:
        self.calls.append(question)
        return FakePipelineResult()


@pytest.fixture
def fake_pipeline() -> FakeEnterpriseRagPipeline:
    return FakeEnterpriseRagPipeline()


@pytest.fixture
def graph_db_path(tmp_path: Path) -> Path:
    path = tmp_path / "banking.db"
    with banking_data.session(path) as conn:
        conn.executemany(
            "INSERT INTO customers (customer_id, display_name) VALUES (?, ?)",
            [("CUST001", "Asha Rao"), ("CUST002", "Vikram Nair")],
        )
        conn.executemany(
            "INSERT INTO accounts (account_id, customer_id, account_type, balance_paise, currency) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("ACC-1", "CUST001", "SAVINGS", 1500000, "INR"),
                ("ACC-2", "CUST002", "SAVINGS", 800000, "INR"),
            ],
        )
        conn.executemany(
            "INSERT INTO transactions (transaction_id, customer_id, account_id, card_id, "
            "amount_paise, currency, merchant, category, txn_date, status) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "TX1002",
                    "CUST001",
                    "ACC-1",
                    None,
                    45000,
                    "INR",
                    "Swiggy",
                    "Food",
                    "2026-07-30",
                    "POSTED",
                ),
                (
                    "TX1007",
                    "CUST001",
                    "ACC-1",
                    None,
                    450000,
                    "INR",
                    "QuickPay",
                    "Shopping",
                    "2026-08-01",
                    "POSTED",
                ),
                (
                    "TX2001",
                    "CUST002",
                    "ACC-2",
                    None,
                    250000,
                    "INR",
                    "BigBasket",
                    "Groceries",
                    "2026-07-29",
                    "POSTED",
                ),
            ],
        )
        conn.commit()
    return path


@pytest.fixture
def cust001_context() -> SecurityContext:
    return SecurityContext(
        user_id="USR-001", role="CUSTOMER", customer_id="CUST001", session_id="S1", request_id="R1"
    )


@pytest.fixture
def cust002_context() -> SecurityContext:
    return SecurityContext(
        user_id="USR-002", role="CUSTOMER", customer_id="CUST002", session_id="S2", request_id="R2"
    )
