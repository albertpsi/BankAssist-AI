from pathlib import Path

import pytest

from bankassist.tools import banking_data


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
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
                ("ACC-1", "CUST001", "SAVINGS", 100000, "INR"),
                ("ACC-2", "CUST002", "SAVINGS", 200000, "INR"),
            ],
        )
        conn.executemany(
            "INSERT INTO transactions (transaction_id, customer_id, account_id, card_id, "
            "amount_paise, currency, merchant, category, txn_date, status) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "TX1",
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
                    "TX2",
                    "CUST002",
                    "ACC-2",
                    None,
                    99999,
                    "INR",
                    "BigBasket",
                    "Groceries",
                    "2026-08-01",
                    "POSTED",
                ),
            ],
        )
        conn.commit()
    return path
