"""Typed input/output for every scoped tool (Lab 4, FR-9)."""

from __future__ import annotations

from pydantic import BaseModel


class Account(BaseModel):
    account_id: str
    account_type: str
    balance_paise: int
    currency: str


class AccountsResult(BaseModel):
    accounts: list[Account]


class Transaction(BaseModel):
    transaction_id: str
    account_id: str
    card_id: str | None
    amount_paise: int
    currency: str
    merchant: str
    category: str
    txn_date: str
    status: str


class TransactionsResult(BaseModel):
    transactions: list[Transaction]


class TransactionDetailResult(BaseModel):
    transaction: Transaction


class EligibilityResult(BaseModel):
    transaction_id: str
    eligible: bool
    reason: str


class CreateDisputeRequest(BaseModel):
    transaction_id: str
    reason: str


class DisputeResult(BaseModel):
    dispute_id: str
    reference: str
    transaction_id: str
    status: str
