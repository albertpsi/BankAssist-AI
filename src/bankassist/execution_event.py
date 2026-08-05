"""ExecutionEvent — the UI-facing "what executed?" model (Lab 4 brief §14, §29).

Deliberately smaller and separate from ``bankassist.tracing.Span``: ``Span`` keeps
recording full request/LLM/retrieval spans exactly as Labs 1-3 left it. This model
exists only so the Streamlit workflow graph and timeline can be built from real
execution, never from parsing the final answer or matching keywords in the question
(Lab 4 brief §22). Lab 6 is expected to extend or correlate this with tracing, not
Lab 4.

No chain-of-thought, no hidden reasoning, and no secrets ever belong in ``summary``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ExecutionEventType(StrEnum):
    SUPERVISOR_STARTED = "SUPERVISOR_STARTED"
    ROUTE_SELECTED = "ROUTE_SELECTED"
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    RAG_STARTED = "RAG_STARTED"
    RAG_COMPLETED = "RAG_COMPLETED"
    INTERRUPT_CREATED = "INTERRUPT_CREATED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    GRAPH_RESUMED = "GRAPH_RESUMED"
    RESPONSE_GENERATED = "RESPONSE_GENERATED"

    # --- Lab 5: guardrail / security events ---
    INPUT_VALIDATION_PASSED = "INPUT_VALIDATION_PASSED"
    INPUT_VALIDATION_BLOCKED = "INPUT_VALIDATION_BLOCKED"
    NEMO_INPUT_RAIL_PASSED = "NEMO_INPUT_RAIL_PASSED"
    NEMO_INPUT_RAIL_BLOCKED = "NEMO_INPUT_RAIL_BLOCKED"
    AUTHORIZATION_CHECK_PASSED = "AUTHORIZATION_CHECK_PASSED"
    AUTHORIZATION_CHECK_BLOCKED = "AUTHORIZATION_CHECK_BLOCKED"
    OWNERSHIP_CHECK_PASSED = "OWNERSHIP_CHECK_PASSED"
    OWNERSHIP_CHECK_BLOCKED = "OWNERSHIP_CHECK_BLOCKED"
    DISPUTE_MUTATION_ALLOWED = "DISPUTE_MUTATION_ALLOWED"
    DISPUTE_MUTATION_BLOCKED = "DISPUTE_MUTATION_BLOCKED"
    NEMO_OUTPUT_RAIL_PASSED = "NEMO_OUTPUT_RAIL_PASSED"
    NEMO_OUTPUT_RAIL_BLOCKED = "NEMO_OUTPUT_RAIL_BLOCKED"
    OUTPUT_PROTECTION_PASSED = "OUTPUT_PROTECTION_PASSED"
    OUTPUT_PROTECTION_REDACTED = "OUTPUT_PROTECTION_REDACTED"

    # --- Lab 7: cost optimization / caching events (ADR-0013) ---
    CACHE_ELIGIBILITY_DECIDED = "CACHE_ELIGIBILITY_DECIDED"
    SEMANTIC_CACHE_HIT = "SEMANTIC_CACHE_HIT"
    SEMANTIC_CACHE_MISS = "SEMANTIC_CACHE_MISS"
    SEMANTIC_CACHE_BYPASSED = "SEMANTIC_CACHE_BYPASSED"
    SEMANTIC_CACHE_STORED = "SEMANTIC_CACHE_STORED"
    EMBEDDING_CACHE_HIT = "EMBEDDING_CACHE_HIT"
    EMBEDDING_CACHE_MISS = "EMBEDDING_CACHE_MISS"
    TOOL_CACHE_HIT = "TOOL_CACHE_HIT"
    TOOL_CACHE_MISS = "TOOL_CACHE_MISS"
    TOOL_CACHE_BYPASSED = "TOOL_CACHE_BYPASSED"


class ExecutionStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    SKIPPED = "SKIPPED"


class ExecutionEvent(BaseModel):
    """One observable step of a single request's graph execution."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: ExecutionEventType
    node_id: str
    node_type: str
    label: str
    status: ExecutionStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: str = ""
