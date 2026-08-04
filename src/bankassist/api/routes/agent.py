"""Multi-agent chat endpoint (Lab 4, FR-19/FR-20)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from bankassist.agents import (
    ExecutionEvent,
    build_graph,
    is_waiting_for_approval,
    resume_graph,
    run_config,
)
from bankassist.api.auth_dependency import require_security_context
from bankassist.api.routes.rag import get_enterprise_pipeline
from bankassist.api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentResumeRequest,
    ExecutionEventSchema,
    TransactionOption,
)
from bankassist.config import Settings
from bankassist.errors import AuthorizationError, NoPendingApprovalError
from bankassist.guardrails.nemo_adapter import NemoGuardrailsAdapter
from bankassist.llm.factory import build_llm_client
from bankassist.security.context import SecurityContext

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_graph(request: Request) -> Any:
    """Build the compiled LangGraph once per app lifetime, reused across requests.

    Cached separately from the RAG pipelines on ``app.state`` — building the graph
    never constructs a second enterprise pipeline; it reuses the one from
    ``bankassist.api.routes.rag``.
    """
    cached = getattr(request.app.state, "agent_graph", None)
    if cached is not None:
        return cached

    settings: Settings = request.app.state.settings
    tracer = request.app.state.tracer
    llm = build_llm_client(settings, tracer)
    pipeline = get_enterprise_pipeline(request)
    nemo = getattr(request.app.state, "nemo_guardrails", None) or NemoGuardrailsAdapter(settings)
    request.app.state.nemo_guardrails = nemo

    graph = build_graph(
        llm=llm, enterprise_pipeline=pipeline, db_path=settings.banking_db_path, nemo=nemo
    )
    request.app.state.agent_graph = graph
    return graph


def _events_schema(events: list[ExecutionEvent]) -> list[ExecutionEventSchema]:
    return [
        ExecutionEventSchema(
            event_type=e.event_type.value,
            node_id=e.node_id,
            node_type=e.node_type,
            label=e.label,
            status=e.status.value,
            timestamp=e.timestamp.isoformat(),
            summary=e.summary,
        )
        for e in events
    ]


def _transaction_options(result: dict[str, Any]) -> list[TransactionOption] | None:
    """Structured choices for the "which transaction?" step of the dispute flow.

    Populated only when the Dispute Agent showed a transaction list this turn and
    has not yet resolved one — never for a turn that already selected, disputed, or
    ruled out a transaction, so the UI doesn't offer stale buttons.
    """
    if result.get("current_agent") != "dispute_agent" or result.get("selected_transaction_id"):
        return None
    transactions = (result.get("tool_results") or {}).get("recent_transactions")
    if not transactions:
        return None
    return [
        TransactionOption(
            transaction_id=t["transaction_id"],
            merchant=t["merchant"],
            amount_rupees=t["amount_paise"] / 100,
            txn_date=t["txn_date"],
        )
        for t in transactions
    ]


def _resolved_context(
    payload_customer_id: str | None, payload_session_id: str, base: SecurityContext
) -> SecurityContext:
    """FR-24/FR-25 at the API boundary: the body's ``customer_id`` is never trusted.

    If the caller supplies one that disagrees with the authenticated context, the
    request is rejected rather than silently corrected.
    """
    if payload_customer_id is not None and payload_customer_id != base.customer_id:
        raise AuthorizationError(
            "The requested customer does not match the authenticated identity.",
            details={"permission": "customer_scope"},
        )
    return base.model_copy(update={"session_id": payload_session_id})


@router.post("/chat", response_model=AgentChatResponse, summary="Talk to the multi-agent assistant")
def chat(
    payload: AgentChatRequest,
    request: Request,
    context: SecurityContext = Depends(require_security_context),
) -> AgentChatResponse:
    context = _resolved_context(payload.customer_id, payload.session_id, context)
    graph = get_agent_graph(request)
    config = run_config(payload.session_id, context)

    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": payload.message}],
            "customer_id": context.customer_id,
            "session_id": payload.session_id,
            "request_id": context.request_id,
        },
        config,
    )

    events = result.get("execution_events", [])
    waiting = "__interrupt__" in result or is_waiting_for_approval(graph, config)

    if waiting:
        return AgentChatResponse(
            answer="This dispute needs your approval before it can be filed.",
            agent=result.get("current_agent") or "dispute_agent",
            session_id=payload.session_id,
            status="waiting_approval",
            approval_required=True,
            sources=result.get("retrieved_sources") or [],
            execution_events=_events_schema(events),
        )

    blocked = result.get("current_agent") == "guardrail"
    return AgentChatResponse(
        answer=result.get("final_answer") or "",
        agent=result.get("current_agent") or "supervisor",
        session_id=payload.session_id,
        status="blocked" if blocked else "completed",
        approval_required=False,
        sources=result.get("retrieved_sources") or [],
        execution_events=_events_schema(events),
        available_transactions=_transaction_options(result),
    )


@router.post(
    "/resume", response_model=AgentChatResponse, summary="Approve or reject a pending dispute"
)
def resume(
    payload: AgentResumeRequest,
    request: Request,
    context: SecurityContext = Depends(require_security_context),
) -> AgentChatResponse:
    graph = get_agent_graph(request)
    config = run_config(payload.session_id, context)

    if not is_waiting_for_approval(graph, config):
        raise NoPendingApprovalError(
            "There is no pending approval on this session.",
            details={"session_id": payload.session_id},
        )

    result = resume_graph(graph, config, approved=payload.approved)
    events = result.get("execution_events", [])

    return AgentChatResponse(
        answer=result.get("final_answer") or "",
        agent=result.get("current_agent") or "dispute_agent",
        session_id=payload.session_id,
        status="completed",
        approval_required=False,
        sources=result.get("retrieved_sources") or [],
        execution_events=_events_schema(events),
    )
