"""Typed LangGraph state (Lab 4, FR-12/FR-28).

Only conversational/orchestration state lives here — checkpointed by LangGraph's
``MemorySaver`` per ``session_id`` (used as the graph ``thread_id``). Per ADR-0010,
``SecurityContext`` (identity, role, the trustworthy ``customer_id``) is deliberately
**not** part of this state: it is passed per-invocation via
``config["configurable"]["security_context"]`` so nothing checkpointable ever holds a
password hash, a raw JWT, or a signing secret. ``customer_id`` appearing below is a
non-secret label copied from the context for display/routing convenience, never the
authority a tool trusts.

No chain-of-thought or hidden-reasoning field exists here (Lab 4 brief §6/§29).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from bankassist.execution_event import ExecutionEvent

Route = Literal["POLICY", "BANKING", "DISPUTE", "CLARIFICATION", "UNSUPPORTED"]


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class BankAssistState(TypedDict, total=False):
    messages: Annotated[list[ChatTurn], operator.add]

    customer_id: str | None
    session_id: str
    request_id: str

    current_agent: str | None
    route: Route | None
    route_confidence: float | None

    tool_calls: Annotated[list[str], operator.add]
    tool_results: dict[str, Any]
    retrieved_sources: list[str]

    selected_transaction_id: str | None
    dispute_eligibility: dict[str, Any] | None

    pending_action: dict[str, Any] | None
    approval_required: bool
    approval_status: Literal["approved", "rejected"] | None
    # Lab 5 replay guard (FR-31 extension): set True the moment an approval is
    # consumed by create_dispute_node/cancel_dispute_node, so a second resume on the
    # same thread can never re-trigger the mutation even if LangGraph's own
    # "no pending interrupt" check were ever bypassed.
    pending_action_consumed: bool

    execution_events: Annotated[list[ExecutionEvent], operator.add]

    final_answer: str | None

    # Lab 5: routing signal for the input-guardrail nodes, distinct from
    # `final_answer`. `final_answer` is per-turn but only reset by `supervisor_node`,
    # which runs *after* the guardrail nodes — using it directly as a "blocked this
    # turn" flag would misread a *previous* turn's completed answer as a fresh
    # block. This field is reset by `input_validation_node` itself, the first node
    # of every turn, so it never carries state across turns.
    guardrail_blocked: bool
