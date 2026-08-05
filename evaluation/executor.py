"""The live executor: runs a `GoldenCase` through the real, running application.

Not exercised by the default `pytest` run — it needs a real OpenAI key, a
seeded banking database, and (optionally) a live Pinecone index, exactly like
running the app for real. It drives the same HTTP boundary a real client
uses (`POST /agent/chat`, `POST /agent/resume`) via FastAPI's `TestClient`,
so it exercises input/output guardrails, routing, tools, RAG, and HITL
identically to production traffic — and, if `AGENTOPS_ENABLED=true`, the run
shows up in the AgentOps dashboard like any other request (Lab 6 §17).

Usage (see `scripts/run_evaluation.py`):

    from evaluation.executor import GraphExecutor
    from evaluation.runner import run_evaluation
    from evaluation.report import render_markdown

    executor = GraphExecutor()
    report = run_evaluation(executor)
    print(render_markdown(report))
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from bankassist.api.app import create_app
from bankassist.config import Settings, get_settings
from bankassist.security.jwt_tokens import issue_token
from evaluation.models import CaseResult, GoldenCase

# The demo dataset's seeded customer (scripts/seed_banking_data.py). Every
# case runs as this customer — cross-customer-access cases reference a
# *different* customer id inside the message text, not a different login.
DEFAULT_CUSTOMER_ID = "CUST001"
DEFAULT_USER_ID = "USR-001"


class GraphExecutor:
    """Executes golden cases against a real, running BankAssist app instance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._app = create_app(self._settings)
        self._client = TestClient(self._app)
        self._token = issue_token(
            settings=self._settings,
            user_id=DEFAULT_USER_ID,
            role="CUSTOMER",
            customer_id=DEFAULT_CUSTOMER_ID,
        )

    def __call__(self, case: GoldenCase) -> CaseResult:
        session_id = f"eval-{case.case_id}-{uuid.uuid4().hex[:8]}"
        headers = {"Authorization": f"Bearer {self._token}"}

        last_body: dict | None = None
        for turn in case.turn_sequence():
            response = self._client.post(
                "/api/v1/agent/chat",
                json={"message": turn, "session_id": session_id},
                headers=headers,
            )
            if response.status_code != 200:
                return CaseResult(
                    case_id=case.case_id,
                    error=f"HTTP {response.status_code}: {response.text[:300]}",
                )
            last_body = response.json()

        assert last_body is not None  # at least one turn always runs

        return _to_case_result(case.case_id, last_body)


# `AgentChatResponse.agent` is `current_agent` — a LangGraph node id
# ("banking_agent", "dispute_agent", ...), not the supervisor's `route` value
# ("BANKING", "DISPUTE", ...) the golden dataset's `expected_route` uses
# (`agents/state.py`/`supervisor.py`). The API never exposes `route` directly,
# so it is reconstructed here. Bug found by inspecting a live evaluation
# report where only `CLARIFICATION`/`UNSUPPORTED` cases passed routing checks
# — the one pair of node ids that happen to equal their route name already.
_AGENT_TO_ROUTE = {
    "policy_agent": "POLICY",
    "banking_agent": "BANKING",
    "dispute_agent": "DISPUTE",
    "clarification": "CLARIFICATION",
    "unsupported": "UNSUPPORTED",
}


def _to_case_result(case_id: str, body: dict) -> CaseResult:
    events = body.get("execution_events", [])
    tools_called = [e["label"] for e in events if e.get("node_type") == "tool"]
    blocked = body.get("status") == "blocked"
    approval_required = bool(body.get("approval_required"))
    agent = body.get("agent")
    route = _AGENT_TO_ROUTE.get(agent, agent)

    # A structural invariant, not a heuristic: `create_dispute` can only run
    # after `POST /agent/resume`. A response that is still `waiting_approval`
    # but already shows a `create_dispute` tool call would be the one genuine
    # violation this metric exists to catch.
    mutation_before_approval = (
        body.get("status") == "waiting_approval" and "create_dispute" in tools_called
    )

    return CaseResult(
        case_id=case_id,
        answer=body.get("answer", ""),
        route=route,
        tools_called=tools_called,
        retrieved_sources=body.get("sources") or [],
        blocked=blocked,
        approval_required=approval_required,
        mutation_occurred_before_approval=bool(mutation_before_approval),
        grounded=True,
    )
