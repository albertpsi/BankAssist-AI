"""LangGraph StateGraph wiring (Lab 4, ADR-0009). Orchestration only.

Every node is a thin adapter: it calls existing first-party business logic
(``EnterpriseRagPipeline``, scoped tools, RBAC) and translates the result into a
state update plus ``ExecutionEvent``s. No SQL, retrieval, or business rule is written
here (ADR-0009's core rule).

``SecurityContext`` is read from ``config["configurable"]["security_context"]`` — it
is never part of the checkpointed ``BankAssistState`` (ADR-0010, FR-28).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from bankassist.agents import banking_agent, dispute_agent, policy_agent, supervisor
from bankassist.agents.state import BankAssistState
from bankassist.errors import AuthorizationError, BankAssistError
from bankassist.execution_event import (
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
)
from bankassist.llm.base import LLMClient
from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline
from bankassist.security.context import SecurityContext
from bankassist.tools import (
    check_dispute_eligibility,
    create_dispute,
    get_recent_transactions,
    get_transaction_details,
)
from bankassist.tools.models import Transaction


def _security_context(config: RunnableConfig) -> SecurityContext:
    context = config.get("configurable", {}).get("security_context")
    if context is None:
        raise BankAssistError("No security context on this graph invocation.")
    return context


def _event(
    event_type: ExecutionEventType,
    node_id: str,
    node_type: str,
    label: str,
    status: ExecutionStatus,
    summary: str = "",
) -> ExecutionEvent:
    return ExecutionEvent(
        event_type=event_type,
        node_id=node_id,
        node_type=node_type,
        label=label,
        status=status,
        summary=summary,
    )


def build_graph(
    *,
    llm: LLMClient,
    enterprise_pipeline: EnterpriseRagPipeline,
    db_path: Path,
):
    """Construct and compile the four-agent StateGraph with an in-memory checkpointer."""

    def supervisor_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        events = [
            _event(
                ExecutionEventType.SUPERVISOR_STARTED,
                "supervisor",
                "agent",
                "Supervisor Agent",
                ExecutionStatus.RUNNING,
                "Classifying intent",
            )
        ]
        history = [m["content"] for m in state.get("messages", [])[-6:-1]]
        message = state["messages"][-1]["content"]
        decision = supervisor.decide_route(llm, message, history)
        events.append(
            _event(
                ExecutionEventType.ROUTE_SELECTED,
                "supervisor",
                "agent",
                "Supervisor Agent",
                ExecutionStatus.COMPLETED,
                f"Route: {decision.route} ({decision.reason})",
            )
        )
        return {
            "route": decision.route,
            "route_confidence": decision.confidence,
            "current_agent": "supervisor",
            # Per-turn output fields, reset at the top of every new user turn so a
            # prior turn's completed answer/approval never leaks into this turn's
            # routing decision (route_after_dispute reads `final_answer` to decide
            # whether the dispute workflow already concluded this turn).
            "final_answer": None,
            "pending_action": None,
            "approval_required": False,
            "approval_status": None,
            "execution_events": events,
        }

    def route_from_supervisor(state: BankAssistState) -> str:
        route = state.get("route") or "UNSUPPORTED"
        return {
            "POLICY": "policy_agent",
            "BANKING": "banking_agent",
            "DISPUTE": "dispute_agent",
            "CLARIFICATION": "clarification",
            "UNSUPPORTED": "unsupported",
        }.get(route, "unsupported")

    def policy_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        events = [
            _event(
                ExecutionEventType.AGENT_STARTED,
                "policy_agent",
                "agent",
                "Policy Agent",
                ExecutionStatus.RUNNING,
            ),
            _event(
                ExecutionEventType.RAG_STARTED,
                "enterprise_rag",
                "rag",
                "Enterprise RAG",
                ExecutionStatus.RUNNING,
            ),
        ]
        question = state["messages"][-1]["content"]
        result = policy_agent.answer_policy_question(enterprise_pipeline, question)
        events.append(
            _event(
                ExecutionEventType.RAG_COMPLETED,
                "enterprise_rag",
                "rag",
                "Enterprise RAG",
                ExecutionStatus.COMPLETED,
                f"{len(result.sources)} source(s) retrieved",
            )
        )
        events.append(
            _event(
                ExecutionEventType.AGENT_COMPLETED,
                "policy_agent",
                "agent",
                "Policy Agent",
                ExecutionStatus.COMPLETED,
            )
        )
        events.append(
            _event(
                ExecutionEventType.RESPONSE_GENERATED,
                "final_response",
                "response",
                "Final Response",
                ExecutionStatus.COMPLETED,
            )
        )
        return {
            "current_agent": "policy_agent",
            "final_answer": result.answer,
            "retrieved_sources": result.sources,
            "messages": [{"role": "assistant", "content": result.answer}],
            "execution_events": events,
        }

    def banking_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        context = _security_context(config)
        events = [
            _event(
                ExecutionEventType.AGENT_STARTED,
                "banking_agent",
                "agent",
                "Banking Agent",
                ExecutionStatus.RUNNING,
            )
        ]
        try:
            result = banking_agent.answer_banking_request(context, db_path)
        except BankAssistError as exc:
            events.append(
                _event(
                    ExecutionEventType.AGENT_COMPLETED,
                    "banking_agent",
                    "agent",
                    "Banking Agent",
                    ExecutionStatus.FAILED,
                    exc.message,
                )
            )
            return {
                "current_agent": "banking_agent",
                "final_answer": "I couldn't retrieve your account information.",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I couldn't retrieve your account information.",
                    }
                ],
                "execution_events": events,
            }

        events.append(
            _event(
                ExecutionEventType.TOOL_COMPLETED,
                "get_recent_transactions",
                "tool",
                "get_recent_transactions",
                ExecutionStatus.COMPLETED,
                f"{len(result.transactions)} transaction(s)",
            )
        )
        events.append(
            _event(
                ExecutionEventType.AGENT_COMPLETED,
                "banking_agent",
                "agent",
                "Banking Agent",
                ExecutionStatus.COMPLETED,
            )
        )
        events.append(
            _event(
                ExecutionEventType.RESPONSE_GENERATED,
                "final_response",
                "response",
                "Final Response",
                ExecutionStatus.COMPLETED,
            )
        )
        return {
            "current_agent": "banking_agent",
            "final_answer": result.answer,
            "messages": [{"role": "assistant", "content": result.answer}],
            "tool_results": {"recent_transactions": [t.model_dump() for t in result.transactions]},
            "execution_events": events,
        }

    def dispute_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:
        context = _security_context(config)
        events = [
            _event(
                ExecutionEventType.AGENT_STARTED,
                "dispute_agent",
                "agent",
                "Dispute Agent",
                ExecutionStatus.RUNNING,
            )
        ]
        message = state["messages"][-1]["content"]

        prior_raw = (state.get("tool_results") or {}).get("recent_transactions") or []
        prior_transactions = [Transaction(**t) for t in prior_raw]

        transaction_id = state.get("selected_transaction_id")
        if transaction_id is None and prior_transactions:
            transaction_id = dispute_agent.resolve_transaction_id(message, prior_transactions)

        if transaction_id is None:
            try:
                txns_result = get_recent_transactions(context, db_path, limit=5)
            except BankAssistError as exc:
                events.append(
                    _event(
                        ExecutionEventType.AGENT_COMPLETED,
                        "dispute_agent",
                        "agent",
                        "Dispute Agent",
                        ExecutionStatus.FAILED,
                        exc.message,
                    )
                )
                return {
                    "execution_events": events,
                    "final_answer": "I couldn't look up your transactions.",
                }

            events.append(
                _event(
                    ExecutionEventType.TOOL_COMPLETED,
                    "get_recent_transactions",
                    "tool",
                    "get_recent_transactions",
                    ExecutionStatus.COMPLETED,
                    f"{len(txns_result.transactions)} transaction(s) shown",
                )
            )
            lines = ["Which transaction don't you recognize? Here are your recent ones:", ""]
            for txn in txns_result.transactions:
                rupees = txn.amount_paise / 100
                lines.append(f"- {txn.txn_date}  {txn.merchant}  ₹{rupees:,.2f}")
            answer = "\n".join(lines)
            events.append(
                _event(
                    ExecutionEventType.AGENT_COMPLETED,
                    "dispute_agent",
                    "agent",
                    "Dispute Agent",
                    ExecutionStatus.COMPLETED,
                    "Awaiting transaction identification",
                )
            )
            return {
                "current_agent": "dispute_agent",
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
                "tool_results": {
                    "recent_transactions": [t.model_dump() for t in txns_result.transactions]
                },
                "execution_events": events,
            }

        try:
            detail = get_transaction_details(context, db_path, transaction_id=transaction_id)
        except BankAssistError as exc:
            events.append(
                _event(
                    ExecutionEventType.AGENT_COMPLETED,
                    "dispute_agent",
                    "agent",
                    "Dispute Agent",
                    ExecutionStatus.FAILED,
                    exc.message,
                )
            )
            return {
                "execution_events": events,
                "final_answer": "That transaction could not be identified for your account.",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "That transaction could not be identified for your account.",
                    }
                ],
            }

        events.append(
            _event(
                ExecutionEventType.TOOL_COMPLETED,
                "get_transaction_details",
                "tool",
                "get_transaction_details",
                ExecutionStatus.COMPLETED,
                f"Selected {detail.transaction.transaction_id}",
            )
        )

        events.append(
            _event(
                ExecutionEventType.RAG_STARTED,
                "enterprise_rag",
                "rag",
                "Enterprise RAG",
                ExecutionStatus.RUNNING,
            )
        )
        policy = policy_agent.answer_policy_question(
            enterprise_pipeline, dispute_agent.DISPUTE_POLICY_QUESTION
        )
        events.append(
            _event(
                ExecutionEventType.RAG_COMPLETED,
                "enterprise_rag",
                "rag",
                "Enterprise RAG",
                ExecutionStatus.COMPLETED,
                f"{len(policy.sources)} source(s) retrieved",
            )
        )

        eligibility = check_dispute_eligibility(context, db_path, transaction_id=transaction_id)
        events.append(
            _event(
                ExecutionEventType.TOOL_COMPLETED,
                "check_dispute_eligibility",
                "tool",
                "check_dispute_eligibility",
                ExecutionStatus.COMPLETED,
                "Eligible" if eligibility.eligible else f"Not eligible: {eligibility.reason}",
            )
        )

        if not eligibility.eligible:
            answer = f"I can't file a dispute for that transaction: {eligibility.reason}"
            events.append(
                _event(
                    ExecutionEventType.AGENT_COMPLETED,
                    "dispute_agent",
                    "agent",
                    "Dispute Agent",
                    ExecutionStatus.COMPLETED,
                )
            )
            return {
                "current_agent": "dispute_agent",
                "selected_transaction_id": transaction_id,
                "dispute_eligibility": eligibility.model_dump(),
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
                "retrieved_sources": policy.sources,
                "execution_events": events,
            }

        events.append(
            _event(
                ExecutionEventType.AGENT_COMPLETED,
                "dispute_agent",
                "agent",
                "Dispute Agent",
                ExecutionStatus.COMPLETED,
                "Eligible — awaiting human approval",
            )
        )
        return {
            "current_agent": "dispute_agent",
            "selected_transaction_id": transaction_id,
            "dispute_eligibility": eligibility.model_dump(),
            "retrieved_sources": policy.sources,
            "pending_action": {
                "action": "create_dispute",
                "transaction_id": transaction_id,
                "reason": "Customer does not recognize this transaction.",
            },
            "approval_required": True,
            "execution_events": events,
        }

    def route_after_dispute(state: BankAssistState) -> str:
        return END if state.get("final_answer") else "prepare_dispute"

    def prepare_dispute_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        pending = state["pending_action"]
        events = [
            _event(
                ExecutionEventType.INTERRUPT_CREATED,
                "human_approval",
                "approval",
                "Human Approval",
                ExecutionStatus.WAITING_APPROVAL,
                f"Awaiting approval to dispute {pending['transaction_id']}",
            )
        ]
        decision = interrupt(
            {
                "action": pending["action"],
                "transaction_id": pending["transaction_id"],
                "reason": pending["reason"],
            }
        )
        approved = bool(decision)
        events.append(
            _event(
                ExecutionEventType.APPROVAL_GRANTED
                if approved
                else ExecutionEventType.APPROVAL_REJECTED,
                "human_approval",
                "approval",
                "Human Approval",
                ExecutionStatus.COMPLETED,
                "Approved" if approved else "Rejected",
            )
        )
        events.append(
            _event(
                ExecutionEventType.GRAPH_RESUMED,
                "human_approval",
                "approval",
                "Human Approval",
                ExecutionStatus.COMPLETED,
                "Graph resumed",
            )
        )
        return {
            "approval_status": "approved" if approved else "rejected",
            "execution_events": events,
        }

    def route_after_approval(state: BankAssistState) -> str:
        return (
            "create_dispute_node"
            if state.get("approval_status") == "approved"
            else "cancel_dispute"
        )

    def create_dispute_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:
        context = _security_context(config)
        pending = state["pending_action"]
        events = [
            _event(
                ExecutionEventType.TOOL_STARTED,
                "create_dispute",
                "tool",
                "create_dispute",
                ExecutionStatus.RUNNING,
            )
        ]
        try:
            result = create_dispute(
                context,
                db_path,
                transaction_id=pending["transaction_id"],
                reason=pending["reason"],
            )
        except (AuthorizationError, BankAssistError) as exc:
            events.append(
                _event(
                    ExecutionEventType.TOOL_COMPLETED,
                    "create_dispute",
                    "tool",
                    "create_dispute",
                    ExecutionStatus.FAILED,
                    exc.message,
                )
            )
            answer = "The dispute could not be created."
            return {
                "final_answer": answer,
                "messages": [{"role": "assistant", "content": answer}],
                "pending_action": None,
                "execution_events": events,
            }

        answer = f"Dispute created: {result.reference}. We'll follow up on this transaction."
        events.append(
            _event(
                ExecutionEventType.TOOL_COMPLETED,
                "create_dispute",
                "tool",
                "create_dispute",
                ExecutionStatus.COMPLETED,
                result.reference,
            )
        )
        events.append(
            _event(
                ExecutionEventType.RESPONSE_GENERATED,
                "final_response",
                "response",
                "Final Response",
                ExecutionStatus.COMPLETED,
            )
        )
        return {
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "pending_action": None,
            "execution_events": events,
        }

    def cancel_dispute_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        answer = "Understood — I've cancelled that dispute request."
        events = [
            _event(
                ExecutionEventType.RESPONSE_GENERATED,
                "final_response",
                "response",
                "Final Response",
                ExecutionStatus.COMPLETED,
                "Dispute cancelled",
            )
        ]
        return {
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "pending_action": None,
            "execution_events": events,
        }

    def clarification_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        answer = (
            "Could you clarify what you'd like help with — a policy question, your "
            "accounts/transactions, or an unrecognized transaction?"
        )
        events = [
            _event(
                ExecutionEventType.RESPONSE_GENERATED,
                "final_response",
                "response",
                "Final Response",
                ExecutionStatus.COMPLETED,
            )
        ]
        return {
            "current_agent": "clarification",
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "execution_events": events,
        }

    def unsupported_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        answer = (
            "I can help with banking policy, your accounts, or disputing a transaction, "
            "but I can't help with that request — and I can't provide personalized "
            "investment or financial advice."
        )
        events = [
            _event(
                ExecutionEventType.RESPONSE_GENERATED,
                "final_response",
                "response",
                "Final Response",
                ExecutionStatus.COMPLETED,
            )
        ]
        return {
            "current_agent": "unsupported",
            "final_answer": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "execution_events": events,
        }

    builder = StateGraph(BankAssistState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("policy_agent", policy_node)
    builder.add_node("banking_agent", banking_node)
    builder.add_node("dispute_agent", dispute_node)
    builder.add_node("prepare_dispute", prepare_dispute_node)
    builder.add_node("create_dispute_node", create_dispute_node)
    builder.add_node("cancel_dispute", cancel_dispute_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("unsupported", unsupported_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "policy_agent": "policy_agent",
            "banking_agent": "banking_agent",
            "dispute_agent": "dispute_agent",
            "clarification": "clarification",
            "unsupported": "unsupported",
        },
    )
    builder.add_conditional_edges(
        "dispute_agent",
        route_after_dispute,
        {"prepare_dispute": "prepare_dispute", END: END},
    )
    builder.add_conditional_edges(
        "prepare_dispute",
        route_after_approval,
        {"create_dispute_node": "create_dispute_node", "cancel_dispute": "cancel_dispute"},
    )
    builder.add_edge("policy_agent", END)
    builder.add_edge("banking_agent", END)
    builder.add_edge("create_dispute_node", END)
    builder.add_edge("cancel_dispute", END)
    builder.add_edge("clarification", END)
    builder.add_edge("unsupported", END)

    return builder.compile(checkpointer=MemorySaver())


def run_config(session_id: str, context: SecurityContext) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id, "security_context": context}}


def is_waiting_for_approval(graph: Any, config: RunnableConfig) -> bool:
    state = graph.get_state(config)
    return "prepare_dispute" in (state.next or ())


def resume_graph(graph: Any, config: RunnableConfig, *, approved: bool) -> dict[str, Any]:
    return graph.invoke(Command(resume=approved), config)
