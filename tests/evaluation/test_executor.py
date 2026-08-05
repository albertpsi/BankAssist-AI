"""`evaluation.executor._to_case_result` — the pure response->CaseResult mapping,
tested without a live app (no network, matches the rest of the suite)."""

from __future__ import annotations

from evaluation.executor import _to_case_result


def test_agent_node_id_maps_to_the_supervisors_route_value() -> None:
    """Regression: `AgentChatResponse.agent` is the LangGraph node id
    ("banking_agent"), not the route enum value ("BANKING") the golden
    dataset's `expected_route` uses — found via a live evaluation report
    where only clarification/unsupported cases passed routing checks."""
    body = {"agent": "banking_agent", "status": "completed", "execution_events": []}
    assert _to_case_result("X", body).route == "BANKING"


def test_all_named_agents_map_to_their_route() -> None:
    for agent, expected_route in [
        ("policy_agent", "POLICY"),
        ("banking_agent", "BANKING"),
        ("dispute_agent", "DISPUTE"),
        ("clarification", "CLARIFICATION"),
        ("unsupported", "UNSUPPORTED"),
    ]:
        body = {"agent": agent, "status": "completed", "execution_events": []}
        assert _to_case_result("X", body).route == expected_route


def test_unknown_agent_value_passes_through_unchanged() -> None:
    body = {"agent": "guardrail", "status": "blocked", "execution_events": []}
    assert _to_case_result("X", body).route == "guardrail"


def test_tools_called_extracted_from_tool_type_events_only() -> None:
    body = {
        "agent": "banking_agent",
        "status": "completed",
        "execution_events": [
            {"node_type": "guardrail", "label": "Authorization"},
            {"node_type": "tool", "label": "get_recent_transactions"},
        ],
    }
    assert _to_case_result("X", body).tools_called == ["get_recent_transactions"]


def test_mutation_before_approval_flagged_only_when_still_waiting() -> None:
    waiting_with_mutation = {
        "agent": "dispute_agent",
        "status": "waiting_approval",
        "execution_events": [{"node_type": "tool", "label": "create_dispute"}],
    }
    assert _to_case_result("X", waiting_with_mutation).mutation_occurred_before_approval is True

    completed_with_mutation = {
        "agent": "dispute_agent",
        "status": "completed",
        "execution_events": [{"node_type": "tool", "label": "create_dispute"}],
    }
    assert (
        _to_case_result("X", completed_with_mutation).mutation_occurred_before_approval is False
    )
