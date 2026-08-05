"""Streamlit "Agentic Assistant" page (Lab 4, §20/§24).

Login -> chat -> a Graphviz workflow diagram + a compact timeline, both rendered
purely from the ``execution_events`` the API returns for the last turn — never from
keyword-matching the question or the answer (Lab 4 brief §22). Human approval for
``create_dispute`` is a real LangGraph interrupt/resume round trip through
``POST /api/v1/agent/resume``, not a same-turn string check.

    streamlit run src/bankassist/ui/agentic_app.py
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import streamlit as st

from bankassist.config import get_settings

st.set_page_config(page_title="BankAssist AI — Agentic Assistant", page_icon="🤖", layout="wide")

settings = get_settings()
BASE_URL = settings.api_base_url.rstrip("/")

# The fixed set of nodes the Lab 4 brief asks the graph to show (§15). Position is
# display order only; each node's colour for *this* request is derived below from
# real ExecutionEvents, never hard-coded per query.
GRAPH_NODES: list[tuple[str, str]] = [
    ("user", "User Request"),
    # --- Lab 5: guarded input boundary ---
    ("input_validation", "Input Validation"),
    ("nemo_input_rail", "NeMo Input Guardrail"),
    ("supervisor", "Supervisor Agent"),
    ("policy_agent", "Policy Agent"),
    ("banking_agent", "Banking Agent"),
    ("dispute_agent", "Dispute Agent"),
    ("banking_agent_authorization", "Authorization"),
    ("dispute_authorization", "Authorization"),
    ("ownership_check", "Ownership"),
    ("enterprise_rag", "Enterprise RAG"),
    ("get_recent_transactions", "get_recent_transactions"),
    ("get_transaction_details", "get_transaction_details"),
    ("check_dispute_eligibility", "check_dispute_eligibility"),
    ("human_approval", "Human Approval"),
    ("dispute_mutation_guard", "Financial Mutation Guard"),
    ("create_dispute", "create_dispute"),
    # --- Lab 5: guarded output boundary ---
    ("nemo_output_rail", "NeMo Output Guardrail"),
    ("output_protection", "PII / Secret Protection"),
    ("final_response", "Final Response"),
]

STATUS_COLOR = {
    "NOT_TRIGGERED": "#e0e0e0",
    "RUNNING": "#fff3b0",
    "COMPLETED": "#b7e4c7",
    "WAITING_APPROVAL": "#ffd6a5",
    "FAILED": "#ffadad",
}


def _node_status(events: list[dict[str, Any]], node_id: str) -> str:
    """The *last* status recorded for ``node_id`` in this turn's event stream."""
    matches = [e for e in events if e["node_id"] == node_id]
    if not matches:
        return "NOT_TRIGGERED"
    return matches[-1]["status"]


def _build_graphviz(events: list[dict[str, Any]]) -> str:
    lines = ["digraph G {", 'rankdir="TB"; node [shape=box, style=filled, fontname="Helvetica"];']
    for node_id, label in GRAPH_NODES:
        status = "COMPLETED" if node_id == "user" and events else _node_status(events, node_id)
        color = STATUS_COLOR.get(status, STATUS_COLOR["NOT_TRIGGERED"])
        suffix = f"\\n[{status}]" if node_id != "user" else ""
        lines.append(f'"{node_id}" [label="{label}{suffix}", fillcolor="{color}"];')

    edges = [
        ("user", "input_validation"),
        ("input_validation", "nemo_input_rail"),
        ("nemo_input_rail", "supervisor"),
        ("supervisor", "policy_agent"),
        ("supervisor", "banking_agent"),
        ("supervisor", "dispute_agent"),
        ("banking_agent", "banking_agent_authorization"),
        ("dispute_agent", "dispute_authorization"),
        ("dispute_agent", "ownership_check"),
        ("policy_agent", "enterprise_rag"),
        ("dispute_agent", "enterprise_rag"),
        ("banking_agent_authorization", "get_recent_transactions"),
        ("dispute_agent", "get_recent_transactions"),
        ("ownership_check", "get_transaction_details"),
        ("dispute_agent", "check_dispute_eligibility"),
        ("dispute_authorization", "human_approval"),
        ("check_dispute_eligibility", "human_approval"),
        ("human_approval", "dispute_mutation_guard"),
        ("dispute_mutation_guard", "create_dispute"),
        ("enterprise_rag", "nemo_output_rail"),
        ("get_recent_transactions", "nemo_output_rail"),
        ("create_dispute", "nemo_output_rail"),
        ("nemo_output_rail", "output_protection"),
        ("output_protection", "final_response"),
    ]
    for src, dst in edges:
        lines.append(f'"{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)


def _login(username: str, password: str) -> dict[str, Any]:
    response = httpx.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _chat(message: str) -> dict[str, Any]:
    response = httpx.post(
        f"{BASE_URL}/api/v1/agent/chat",
        json={"message": message, "session_id": st.session_state.session_id},
        headers=_auth_headers(),
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _resume(approved: bool) -> dict[str, Any]:
    response = httpx.post(
        f"{BASE_URL}/api/v1/agent/resume",
        json={"session_id": st.session_state.session_id, "approved": approved},
        headers=_auth_headers(),
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _cache_stats() -> dict[str, Any] | None:
    """`GET /api/v1/cache/stats` (Lab 7). Never blocks the UI on failure — the
    optimization summary just stays hidden for that turn."""
    try:
        response = httpx.get(
            f"{BASE_URL}/api/v1/cache/stats", headers=_auth_headers(), timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


# Demo-scale cost/latency assumptions for the "estimated saved" figures below —
# not measured production numbers (Lab 7 plan §7). Kept as named constants, not
# inline literals, so the assumption is visible and easy to challenge/update.
_ASSUMED_GENERATION_LATENCY_MS = 6_000  # a full RAG + LLM generation turn, NFR-3
_ASSUMED_CACHE_HIT_LATENCY_MS = 300
_ASSUMED_GENERATION_COST_USD = 0.0015  # ~gpt-4o-mini, one policy-question turn
_ASSUMED_EMBEDDING_COST_USD = 0.00002  # text-embedding-3-small, one query


def _render_optimization_summary(events: list[dict[str, Any]]) -> None:
    """Lab 7 amendment #6: cache decisions, LLM skipped/executed, and estimated
    latency/cost saved for the last turn, plus session-cumulative cache stats."""
    st.subheader("⚡ Optimization Summary (Lab 7)")

    cache_events = [e for e in events if e.get("node_type") == "cache"]
    if not cache_events:
        st.caption("No cache-eligible step ran this turn (e.g. a banking/dispute request).")
    for event in cache_events:
        event_type = event["event_type"]
        if "HIT" in event_type:
            icon = "✅"
        elif "STORE" in event_type:
            icon = "➖"
        else:
            icon = "❌"
        st.markdown(f"{icon} **{event['label']}** — {event_type}")
        if event.get("summary"):
            st.caption(event["summary"])

    semantic_hit = any(e["event_type"] == "SEMANTIC_CACHE_HIT" for e in cache_events)
    llm_skipped = semantic_hit
    status = "⏭️ Skipped (served from cache)" if llm_skipped else "▶️ Executed"
    st.markdown(f"**LLM generation:** {status}")

    if semantic_hit:
        latency_saved = _ASSUMED_GENERATION_LATENCY_MS - _ASSUMED_CACHE_HIT_LATENCY_MS
        st.markdown(
            f"**Estimated latency saved this turn:** ~{latency_saved:,} ms "
            f"(demo assumption: a full RAG+LLM turn ≈ {_ASSUMED_GENERATION_LATENCY_MS:,} ms)"
        )
        st.markdown(f"**Estimated cost saved this turn:** ~${_ASSUMED_GENERATION_COST_USD:.4f}")

    stats = _cache_stats()
    if stats is None:
        return
    st.caption("Session-cumulative cache statistics (`GET /api/v1/cache/stats`):")
    cols = st.columns(4)
    cols[0].metric("Semantic hits", stats["semantic_hits"])
    cols[1].metric("Embedding hits", stats["embedding_hits"])
    cols[2].metric("Tool hits", stats["tool_hits"])
    cols[3].metric("Avg Redis latency (ms)", round(stats["average_redis_latency_ms"], 2))
    estimated_saved = (
        stats["semantic_hits"] * _ASSUMED_GENERATION_COST_USD
        + stats["embedding_hits"] * _ASSUMED_EMBEDDING_COST_USD
    )
    st.caption(f"Estimated cumulative cost saved (demo assumptions): ~${estimated_saved:.4f}")


def _render_login() -> None:
    st.title("BankAssist AI — Agentic Assistant")
    st.caption(
        "Lab 4: Supervisor → Policy / Banking / Dispute agents, LangGraph-orchestrated. "
        "Lab 5: NeMo AI-semantic guardrails + deterministic security boundary added."
    )
    st.subheader("Sign in")
    with st.form("login"):
        username = st.text_input("Username", value="customer1")
        password = st.text_input("Password", type="password", value="Demo@Pass123")
        submitted = st.form_submit_button("Log in")
    if submitted:
        try:
            payload = _login(username, password)
        except httpx.HTTPStatusError:
            st.error("Invalid username or password.")
            return
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API: {exc}")
            return
        st.session_state.token = payload["access_token"]
        st.session_state.role = payload["role"]
        st.session_state.customer_id = payload["customer_id"]
        st.session_state.username = username
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.session_state.last_events = []
        st.session_state.waiting_approval = False
        st.session_state.available_transactions = []
        st.rerun()


def _render_assistant() -> None:
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.title("BankAssist AI — Agentic Assistant")
        st.caption(
            f"Logged in as **{st.session_state.username}**  |  "
            f"Role: **{st.session_state.role}**  |  "
            f"Customer: **{st.session_state.customer_id or '—'}**"
        )
    with top_right:
        if st.button("Log out"):
            for key in (
                "token",
                "role",
                "customer_id",
                "username",
                "messages",
                "last_events",
                "session_id",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    chat_col, graph_col = st.columns([1, 1])

    with chat_col:
        st.subheader("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.get("waiting_approval"):
            st.info("A dispute is awaiting your approval before it can be filed.")
            approve_col, reject_col = st.columns(2)
            if approve_col.button("✅ APPROVE", use_container_width=True):
                result = _resume(True)
                _apply_result(result)
                st.rerun()
            if reject_col.button("❌ REJECT", use_container_width=True):
                result = _resume(False)
                _apply_result(result)
                st.rerun()
        else:
            choices = st.session_state.get("available_transactions") or []
            if choices:
                st.caption("Select the transaction you don't recognize:")
                for choice in choices:
                    amount = choice["amount_rupees"]
                    label = f"{choice['txn_date']}  {choice['merchant']}  ₹{amount:,.2f}"
                    if st.button(
                        label, key=f"txn-{choice['transaction_id']}", use_container_width=True
                    ):
                        _send_message(f"The ₹{choice['amount_rupees']:,.2f} one.")

            question = st.chat_input("Ask about policy, your accounts, or a transaction...")
            if question:
                _send_message(question)

    with graph_col:
        st.subheader("Agent Execution Graph")
        events = st.session_state.get("last_events", [])
        st.graphviz_chart(_build_graphviz(events))

        st.subheader("Execution Timeline")
        if not events:
            st.caption("No request executed yet.")
        for i, event in enumerate(events, start=1):
            st.markdown(f"**{i}. {event['label']}** — {event['status']}")
            if event["summary"]:
                st.caption(event["summary"])

        sources = st.session_state.get("last_sources", [])
        if sources:
            st.subheader("RAG Sources")
            st.write(", ".join(sources))

        st.divider()
        _render_optimization_summary(events)


def _send_message(text: str) -> None:
    """Shared path for both free-text chat input and a clicked transaction button."""
    st.session_state.messages.append({"role": "user", "content": text})
    try:
        result = _chat(text)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("error", {}).get("message", exc.response.text)
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Request failed: {detail}"}
        )
    else:
        _apply_result(result)
    st.rerun()


def _apply_result(result: dict[str, Any]) -> None:
    st.session_state.last_events = result.get("execution_events", [])
    st.session_state.last_sources = result.get("sources", [])
    st.session_state.waiting_approval = result.get("status") == "waiting_approval"
    st.session_state.available_transactions = result.get("available_transactions") or []
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


if "token" not in st.session_state:
    _render_login()
else:
    _render_assistant()
