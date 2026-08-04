"""Seed deterministic synthetic banking + user data for Lab 4 (FR-8/FR-21).

Idempotent: re-running clears and re-inserts the seeded rows. All data is
synthetic — no real names, no real PANs (CLAUDE.md §7). Card numbers are
non-Luhn-valid, always masked.

    python scripts/seed_banking_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankassist.config import get_settings
from bankassist.security.passwords import hash_password
from bankassist.tools import banking_data

CUSTOMERS = [
    ("CUST001", "Asha Rao"),
    ("CUST002", "Vikram Nair"),
]

ACCOUNTS = [
    ("ACC-C001-01", "CUST001", "SAVINGS", 15_25000, "INR"),
    ("ACC-C002-01", "CUST002", "SAVINGS", 8_40000, "INR"),
]

CARDS = [
    ("CARD-C001-01", "CUST001", "ACC-C001-01", "****-****-****-4321", "DEBIT", "ACTIVE"),
    ("CARD-C002-01", "CUST002", "ACC-C002-01", "****-****-****-8765", "DEBIT", "ACTIVE"),
]

# amount_paise: ₹4,500.00 = 450000 paise — the amount the multi-turn demo references.
TRANSACTIONS = [
    (
        "TX1001",
        "CUST001",
        "ACC-C001-01",
        "CARD-C001-01",
        120000,
        "INR",
        "Amazon.in",
        "Shopping",
        "2026-07-28",
        "POSTED",
    ),
    (
        "TX1002",
        "CUST001",
        "ACC-C001-01",
        "CARD-C001-01",
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
        "ACC-C001-01",
        "CARD-C001-01",
        450000,
        "INR",
        "QuickPay Electronics",
        "Shopping",
        "2026-08-01",
        "POSTED",
    ),
    (
        "TX1010",
        "CUST001",
        "ACC-C001-01",
        "CARD-C001-01",
        75000,
        "INR",
        "IRCTC",
        "Travel",
        "2026-08-02",
        "POSTED",
    ),
    (
        "TX2001",
        "CUST002",
        "ACC-C002-01",
        "CARD-C002-01",
        250000,
        "INR",
        "BigBasket",
        "Groceries",
        "2026-07-29",
        "POSTED",
    ),
    (
        "TX2002",
        "CUST002",
        "ACC-C002-01",
        "CARD-C002-01",
        999900,
        "INR",
        "MakeMyTrip",
        "Travel",
        "2026-08-01",
        "POSTED",
    ),
]

USERS = [
    # (id, username, password, customer_id, role)
    ("USR-001", "customer1", "Demo@Pass123", "CUST001", "CUSTOMER"),
    ("USR-002", "customer2", "Demo@Pass123", "CUST002", "CUSTOMER"),
    ("USR-003", "support1", "Demo@Pass123", None, "SUPPORT_AGENT"),
    ("USR-004", "admin1", "Demo@Pass123", None, "ADMIN"),
]


def seed() -> None:
    settings = get_settings()
    with banking_data.session(settings.banking_db_path) as conn:
        conn.executescript(
            "DELETE FROM disputes; DELETE FROM transactions; DELETE FROM cards; "
            "DELETE FROM accounts; DELETE FROM users; DELETE FROM customers;"
        )
        conn.executemany(
            "INSERT INTO customers (customer_id, display_name) VALUES (?, ?)", CUSTOMERS
        )
        conn.executemany(
            "INSERT INTO accounts (account_id, customer_id, account_type, balance_paise, currency) "
            "VALUES (?, ?, ?, ?, ?)",
            ACCOUNTS,
        )
        conn.executemany(
            "INSERT INTO cards (card_id, customer_id, account_id, masked_number, card_type, "
            "status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            CARDS,
        )
        conn.executemany(
            "INSERT INTO transactions (transaction_id, customer_id, account_id, card_id, "
            "amount_paise, currency, merchant, category, txn_date, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            TRANSACTIONS,
        )
        conn.executemany(
            "INSERT INTO users (id, username, password_hash, customer_id, role, is_active, "
            "created_at) "
            "VALUES (?, ?, ?, ?, ?, 1, datetime('now'))",
            [
                (uid, username, hash_password(password), customer_id, role)
                for uid, username, password, customer_id, role in USERS
            ],
        )
        conn.commit()
    print(
        f"Seeded {settings.banking_db_path} with {len(CUSTOMERS)} customers, "
        f"{len(TRANSACTIONS)} transactions, {len(USERS)} users."
    )
    print(
        "Demo credentials (username / password): "
        + ", ".join(f"{username} / {password}" for _, username, password, _, _ in USERS)
    )


if __name__ == "__main__":
    seed()
