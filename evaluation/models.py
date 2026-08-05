"""Pydantic models for the golden dataset, case execution, and the report.

Every boundary object gets a model (CLAUDE.md §5), same as the rest of the
codebase — the evaluation subsystem is not exempt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "policy_rag",
    "banking_agent",
    "dispute_agent",
    "routing",
    "multi_turn",
    "security_guardrails",
    "must_allow",
    "must_block",
]


class GoldenCase(BaseModel):
    """One row of the golden evaluation dataset."""

    case_id: str
    category: Category
    description: str = ""

    # A single-turn case supplies `query`; a multi-turn case supplies `turns`
    # (each turn is a user message, later turns building on earlier state).
    query: str | None = None
    turns: list[str] | None = None

    # --- Expectations (only the ones relevant to `category` are set) ---
    expected_sources: list[str] = Field(default_factory=list)
    expected_route: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    unexpected_tools: list[str] = Field(default_factory=list)
    expected_refusal: bool = False
    forbidden_terms: list[str] = Field(default_factory=list)
    approval_required: bool | None = None
    mutation_before_approval: bool | None = None  # must always be False when set
    must_block: bool = False  # True for an attack case; False for a legitimate one

    def turn_sequence(self) -> list[str]:
        if self.turns:
            return self.turns
        if self.query:
            return [self.query]
        raise ValueError(f"{self.case_id}: neither `query` nor `turns` is set.")


class CaseResult(BaseModel):
    """What actually happened when a `GoldenCase` was executed.

    Populated by an executor (``evaluation.executor`` for a real/live run, a
    stub in tests) — the runner and metrics never talk to the graph directly.
    """

    case_id: str
    answer: str = ""
    route: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    blocked: bool = False
    approval_required: bool = False
    mutation_occurred_before_approval: bool = False
    grounded: bool = True
    latency_ms: float = 0.0
    agentops_trace_id: str | None = None
    error: str | None = None


class CaseScore(BaseModel):
    """Pass/fail + the metric values that produced it, for one case."""

    case_id: str
    category: Category
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
    agentops_trace_id: str | None = None


class EvaluationReport(BaseModel):
    """The compact report described in Lab 6 requirements §18."""

    total_cases: int
    passed: int
    failed: int

    routing_accuracy: float | None = None
    tool_selection_accuracy: float | None = None
    retrieval_hit_at_k: float | None = None
    retrieval_mrr: float | None = None
    citation_accuracy: float | None = None
    attack_block_rate: float | None = None
    legitimate_allow_rate: float | None = None
    average_latency_ms: float | None = None

    case_scores: list[CaseScore] = Field(default_factory=list)
