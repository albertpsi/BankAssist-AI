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
from bankassist.caching.semantic_cache import SemanticCache
from bankassist.errors import AuthorizationError, BankAssistError
from bankassist.execution_event import (
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
)
from bankassist.guardrails import GuardrailResult
from bankassist.guardrails.input_validation import validate_message_shape
from bankassist.guardrails.masking import mask_sensitive_identifiers
from bankassist.guardrails.nemo_adapter import NemoGuardrailsAdapter
from bankassist.guardrails.redaction import redact
from bankassist.guardrails.tool_authorization import (
    check_dispute_mutation_allowed,
    check_permission,
)
from bankassist.llm.base import LLMClient
from bankassist.observability import run as observability_run
from bankassist.observability import update_metadata
from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline
from bankassist.security.authorize import Permission
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


def _guardrail_event(
    result: GuardrailResult,
    *,
    node_id: str,
    label: str,
    passed_type: ExecutionEventType,
    blocked_type: ExecutionEventType,
) -> ExecutionEvent:
    """Project a ``GuardrailResult`` into an ``ExecutionEvent`` (Lab 5 §8/§11).

    Only the user-safe ``reason`` reaches ``summary`` — ``internal_reason`` never
    leaves this process.
    """
    if result.allowed:
        return _event(
            passed_type, node_id, "guardrail", label, ExecutionStatus.COMPLETED, result.reason
        )
    return _event(blocked_type, node_id, "guardrail", label, ExecutionStatus.FAILED, result.reason)


def build_graph(
    *,
    llm: LLMClient,
    enterprise_pipeline: EnterpriseRagPipeline,
    db_path: Path,
    nemo: NemoGuardrailsAdapter,
    semantic_cache: SemanticCache | None = None,
):
    """Construct and compile the guarded StateGraph with an in-memory checkpointer.

    Lab 5 adds four boundary nodes around the Lab 4 agent graph: deterministic input
    validation and the NeMo input rail run before the supervisor; the NeMo output
    rail and deterministic PII/secret protection run after every terminal agent node,
    funnelled through a single ``output_guardrails`` node before ``END`` (ADR-0011).

    Lab 7 (ADR-0013): ``semantic_cache`` is optional — with none given, the policy
    node behaves exactly as Labs 4-6 shipped. When given, the policy node consults
    it before running the enterprise RAG pipeline, and stores the answer after,
    subject to ADR-0006's eligibility rule (never touched for BANKING/DISPUTE,
    which always invoke a customer-scoped tool).
    """

    def input_validation_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        # First node of every turn (Lab 5): clears the previous turn's leftover
        # final_answer/guardrail_blocked before anything else runs, so a completed
        # prior answer is never mistaken for "this turn was blocked".
        message = state["messages"][-1]["content"]
        result = validate_message_shape(message)
        event = _guardrail_event(
            result,
            node_id="input_validation",
            label="Input Validation",
            passed_type=ExecutionEventType.INPUT_VALIDATION_PASSED,
            blocked_type=ExecutionEventType.INPUT_VALIDATION_BLOCKED,
        )
        if result.allowed:
            return {"execution_events": [event], "final_answer": None, "guardrail_blocked": False}
        return {
            "execution_events": [event],
            "final_answer": result.reason,
            "messages": [{"role": "assistant", "content": result.reason}],
            "guardrail_blocked": True,
        }

    def route_after_input_validation(state: BankAssistState) -> str:
        return "blocked_response" if state.get("guardrail_blocked") else "nemo_input_rail"

    def nemo_input_rail_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        message = state["messages"][-1]["content"]
        result = nemo.check_input(message)
        event = _guardrail_event(
            result,
            node_id="nemo_input_rail",
            label="NeMo Input Guardrail",
            passed_type=ExecutionEventType.NEMO_INPUT_RAIL_PASSED,
            blocked_type=ExecutionEventType.NEMO_INPUT_RAIL_BLOCKED,
        )
        if result.allowed:
            return {"execution_events": [event]}
        return {
            "execution_events": [event],
            "final_answer": result.reason,
            "messages": [{"role": "assistant", "content": result.reason}],
            "guardrail_blocked": True,
        }

    def route_after_nemo_input_rail(state: BankAssistState) -> str:
        return "blocked_response" if state.get("guardrail_blocked") else "supervisor"

    def blocked_response_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        # The refusal was already generated by whichever guardrail blocked this
        # request; nothing downstream (supervisor, agents, tools) ever ran.
        return {"current_agent": "guardrail"}

    def output_guardrails_node(state: BankAssistState, config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
        answer = state.get("final_answer")
        if not answer:
            return {}

        events: list[ExecutionEvent] = []
        nemo_result = nemo.check_output(answer)
        events.append(
            _guardrail_event(
                nemo_result,
                node_id="nemo_output_rail",
                label="NeMo Output Guardrail",
                passed_type=ExecutionEventType.NEMO_OUTPUT_RAIL_PASSED,
                blocked_type=ExecutionEventType.NEMO_OUTPUT_RAIL_BLOCKED,
            )
        )
        if not nemo_result.allowed:
            safe_answer = "This response could not be delivered as generated."
            events.append(
                _event(
                    ExecutionEventType.OUTPUT_PROTECTION_PASSED,
                    "output_protection",
                    "guardrail",
                    "PII / Secret Protection",
                    ExecutionStatus.SKIPPED,
                    "Skipped — response already blocked upstream.",
                )
            )
            return {
                "final_answer": safe_answer,
                "messages": [{"role": "assistant", "content": safe_answer}],
                "execution_events": events,
            }

        protected = mask_sensitive_identifiers(redact(answer))
        protection_event = _event(
            ExecutionEventType.OUTPUT_PROTECTION_REDACTED
            if protected != answer
            else ExecutionEventType.OUTPUT_PROTECTION_PASSED,
            "output_protection",
            "guardrail",
            "PII / Secret Protection",
            ExecutionStatus.COMPLETED,
            "Redacted sensitive content." if protected != answer else "No sensitive content found.",
        )
        events.append(protection_event)
        if protected == answer:
            return {"execution_events": events}
        return {
            "final_answer": protected,
            "messages": [{"role": "assistant", "content": protected}],
            "execution_events": events,
        }

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
        # Named AgentOps operation span: the routing decision itself is not
        # visible to automatic LangGraph instrumentation as a distinct step
        # (Lab 6 requirements §5/§6).
        decision = observability_run(
            "operation", "supervisor.route", supervisor.decide_route, llm, message, history
        )
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
            "pending_action_consumed": False,
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
            )
        ]
        question = state["messages"][-1]["content"]

        if semantic_cache is not None:
            cache_result = observability_run(
                "operation",
                "semantic_cache.lookup",
                semantic_cache.lookup,
                question,
                route="POLICY",
            )
            events.append(
                _event(
                    ExecutionEventType.SEMANTIC_CACHE_HIT
                    if cache_result.hit
                    else ExecutionEventType.SEMANTIC_CACHE_MISS,
                    "semantic_cache",
                    "cache",
                    "Semantic Cache",
                    ExecutionStatus.COMPLETED,
                    cache_result.reason
                    + (
                        f" (similarity={cache_result.similarity:.3f}, source={cache_result.source})"
                        if cache_result.hit
                        else ""
                    ),
                )
            )
            update_metadata(
                semantic_cache_event="hit" if cache_result.hit else "miss",
                semantic_cache_eligibility=cache_result.eligibility.value,
            )
            if cache_result.hit:
                events.append(
                    _event(
                        ExecutionEventType.AGENT_COMPLETED,
                        "policy_agent",
                        "agent",
                        "Policy Agent",
                        ExecutionStatus.COMPLETED,
                        "Served from semantic cache — RAG and LLM generation skipped.",
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
                    "final_answer": cache_result.response,
                    "retrieved_sources": [],
                    "messages": [{"role": "assistant", "content": cache_result.response}],
                    "execution_events": events,
                }

        events.append(
            _event(
                ExecutionEventType.RAG_STARTED,
                "enterprise_rag",
                "rag",
                "Enterprise RAG",
                ExecutionStatus.RUNNING,
            )
        )
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

        if semantic_cache is not None:
            eligibility = observability_run(
                "operation",
                "semantic_cache.store",
                semantic_cache.store,
                question,
                result.answer,
                route="POLICY",
                customer_scoped_tool_invoked=False,
            )
            events.append(
                _event(
                    ExecutionEventType.SEMANTIC_CACHE_STORED,
                    "semantic_cache",
                    "cache",
                    "Semantic Cache",
                    ExecutionStatus.COMPLETED,
                    f"Eligibility: {eligibility.value}",
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
        # Lab 5: explicit tool-authorization boundary, independent of the fact that
        # the supervisor routed here — routing is not authorization.
        auth_result = check_permission(context, Permission.VIEW_OWN_TRANSACTIONS)
        events.append(
            _guardrail_event(
                auth_result,
                node_id="banking_agent_authorization",
                label="Authorization",
                passed_type=ExecutionEventType.AUTHORIZATION_CHECK_PASSED,
                blocked_type=ExecutionEventType.AUTHORIZATION_CHECK_BLOCKED,
            )
        )
        if not auth_result.allowed:
            return {
                "current_agent": "banking_agent",
                "final_answer": auth_result.reason,
                "messages": [{"role": "assistant", "content": auth_result.reason}],
                "execution_events": events,
            }
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
                txns_result = observability_run(
                    "tool",
                    "get_recent_transactions",
                    get_recent_transactions,
                    context,
                    db_path,
                    limit=5,
                )
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
            detail = observability_run(
                "tool",
                "get_transaction_details",
                get_transaction_details,
                context,
                db_path,
                transaction_id=transaction_id,
            )
        except AuthorizationError as exc:
            # Cross-customer access attempt (AC-8/AC-9): the row exists but belongs to
            # another customer. No transaction data is ever included in the answer.
            events.append(
                _event(
                    ExecutionEventType.OWNERSHIP_CHECK_BLOCKED,
                    "ownership_check",
                    "guardrail",
                    "Ownership",
                    ExecutionStatus.FAILED,
                    "That transaction does not belong to your account.",
                )
            )
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
                ExecutionEventType.OWNERSHIP_CHECK_PASSED,
                "ownership_check",
                "guardrail",
                "Ownership",
                ExecutionStatus.COMPLETED,
                "Customer ownership verified.",
            )
        )
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

        eligibility = observability_run(
            "tool",
            "check_dispute_eligibility",
            check_dispute_eligibility,
            context,
            db_path,
            transaction_id=transaction_id,
        )
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

        # Lab 5: explicit authorization boundary before a mutation is even proposed for
        # approval — routing (supervisor -> dispute_agent) is not authorization.
        auth_result = check_permission(
            context, Permission.CREATE_OWN_DISPUTE, resource_customer_id=context.customer_id
        )
        events.append(
            _guardrail_event(
                auth_result,
                node_id="dispute_authorization",
                label="Authorization",
                passed_type=ExecutionEventType.AUTHORIZATION_CHECK_PASSED,
                blocked_type=ExecutionEventType.AUTHORIZATION_CHECK_BLOCKED,
            )
        )
        if not auth_result.allowed:
            events.append(
                _event(
                    ExecutionEventType.AGENT_COMPLETED,
                    "dispute_agent",
                    "agent",
                    "Dispute Agent",
                    ExecutionStatus.FAILED,
                )
            )
            return {
                "current_agent": "dispute_agent",
                "final_answer": auth_result.reason,
                "messages": [{"role": "assistant", "content": auth_result.reason}],
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
            "pending_action_consumed": False,
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
        # HITL boundary (Lab 6 requirements §5/§6): `interrupt()` suspends this
        # graph invocation entirely — resume happens on a later, separate
        # invocation — so it cannot be wrapped as a single span the way a tool
        # call can. Mark the pause and the eventual decision on the trace
        # metadata instead, so both are visible even though they occur on
        # different invocations of the same LangGraph thread.
        update_metadata(hitl_status="waiting_approval", transaction_id=pending["transaction_id"])
        decision = interrupt(
            {
                "action": pending["action"],
                "transaction_id": pending["transaction_id"],
                "reason": pending["reason"],
            }
        )
        approved = bool(decision)
        update_metadata(hitl_status="approved" if approved else "rejected")
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
        pending = state.get("pending_action")
        events: list[ExecutionEvent] = []

        # Lab 5 §6: the full financial-mutation invariant, independent of and in
        # addition to `create_dispute()`'s own authorize()/eligibility check. Blocks
        # execution before approval, after rejection, and on replay of an
        # already-consumed approval.
        mutation_check = check_dispute_mutation_allowed(
            approval_status=state.get("approval_status"),
            pending_action=pending,
            already_consumed=bool(state.get("pending_action_consumed")),
        )
        events.append(
            _guardrail_event(
                mutation_check,
                node_id="dispute_mutation_guard",
                label="Financial Mutation Guard",
                passed_type=ExecutionEventType.DISPUTE_MUTATION_ALLOWED,
                blocked_type=ExecutionEventType.DISPUTE_MUTATION_BLOCKED,
            )
        )
        if not mutation_check.allowed:
            return {
                "final_answer": "The dispute could not be created.",
                "messages": [{"role": "assistant", "content": "The dispute could not be created."}],
                "pending_action": None,
                "pending_action_consumed": True,
                "execution_events": events,
            }

        events.append(
            _event(
                ExecutionEventType.TOOL_STARTED,
                "create_dispute",
                "tool",
                "create_dispute",
                ExecutionStatus.RUNNING,
            )
        )
        try:
            result = observability_run(
                "tool",
                "create_dispute",
                create_dispute,
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
                "pending_action_consumed": True,
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
            "pending_action_consumed": True,
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
            "pending_action_consumed": True,
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
    builder.add_node("input_validation", input_validation_node)
    builder.add_node("nemo_input_rail", nemo_input_rail_node)
    builder.add_node("blocked_response", blocked_response_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("policy_agent", policy_node)
    builder.add_node("banking_agent", banking_node)
    builder.add_node("dispute_agent", dispute_node)
    builder.add_node("prepare_dispute", prepare_dispute_node)
    builder.add_node("create_dispute_node", create_dispute_node)
    builder.add_node("cancel_dispute", cancel_dispute_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("unsupported", unsupported_node)
    builder.add_node("output_guardrails", output_guardrails_node)

    # --- Input boundary: deterministic checks before probabilistic ones (ADR-0011) ---
    builder.add_edge(START, "input_validation")
    builder.add_conditional_edges(
        "input_validation",
        route_after_input_validation,
        {"blocked_response": "blocked_response", "nemo_input_rail": "nemo_input_rail"},
    )
    builder.add_conditional_edges(
        "nemo_input_rail",
        route_after_nemo_input_rail,
        {"blocked_response": "blocked_response", "supervisor": "supervisor"},
    )
    builder.add_edge("blocked_response", END)

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
        {"prepare_dispute": "prepare_dispute", END: "output_guardrails"},
    )
    builder.add_conditional_edges(
        "prepare_dispute",
        route_after_approval,
        {"create_dispute_node": "create_dispute_node", "cancel_dispute": "cancel_dispute"},
    )

    # --- Output boundary: every terminal agent node funnels through the same
    # NeMo output rail + deterministic redaction/masking pass before END ---
    builder.add_edge("policy_agent", "output_guardrails")
    builder.add_edge("banking_agent", "output_guardrails")
    builder.add_edge("create_dispute_node", "output_guardrails")
    builder.add_edge("cancel_dispute", "output_guardrails")
    builder.add_edge("clarification", "output_guardrails")
    builder.add_edge("unsupported", "output_guardrails")
    builder.add_edge("output_guardrails", END)

    return builder.compile(checkpointer=MemorySaver())


def run_config(session_id: str, context: SecurityContext) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id, "security_context": context}}


def is_waiting_for_approval(graph: Any, config: RunnableConfig) -> bool:
    state = graph.get_state(config)
    return "prepare_dispute" in (state.next or ())


def resume_graph(graph: Any, config: RunnableConfig, *, approved: bool) -> dict[str, Any]:
    return graph.invoke(Command(resume=approved), config)
