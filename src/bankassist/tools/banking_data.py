"""Synthetic banking SQLite schema and connection helper (Lab 4, FR-8/FR-21).

This module owns the schema and a plain connection factory only. It contains no
authorization logic and no scoping — every caller (tools, auth) is responsible for
applying its own ``customer_id``/``role`` filtering. See ``bankassist.tools`` for the
scoped tool functions that are the only sanctioned way agents touch this data, and
``bankassist.security`` for authorization.

All data is synthetic (CLAUDE.md §7): no real names, no real PANs, no real PII.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_type TEXT NOT NULL,
    balance_paise INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR'
);

CREATE TABLE IF NOT EXISTS cards (
    card_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    masked_number TEXT NOT NULL,
    card_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    card_id TEXT REFERENCES cards(card_id),
    amount_paise INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    merchant TEXT NOT NULL,
    category TEXT NOT NULL,
    txn_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'POSTED'
);

CREATE TABLE IF NOT EXISTS disputes (
    dispute_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    reference TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    customer_id TEXT REFERENCES customers(customer_id),
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with row access by column name and FKs enforced."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create every table if it does not already exist. Idempotent."""
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def session(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Short-lived connection for a single tool call."""
    conn = connect(db_path)
    try:
        initialize_schema(conn)
        yield conn
    finally:
        conn.close()
