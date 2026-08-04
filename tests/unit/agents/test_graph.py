from bankassist.agents.graph import build_graph, is_waiting_for_approval, resume_graph, run_config
from bankassist.llm.stub import StubLLMClient
from bankassist.tools import banking_data


def _node_ids(events) -> set[str]:
    return {e.node_id for e in events}


def test_policy_query_routes_through_policy_agent_and_rag_only(
    graph_db_path, fake_pipeline, cust001_context, fake_nemo
):
    llm = StubLLMClient(['{"route": "POLICY", "confidence": 0.9, "reason": "KYC"}'])
    graph = build_graph(
        llm=llm, enterprise_pipeline=fake_pipeline, db_path=graph_db_path, nemo=fake_nemo
    )
    config = run_config("SESS-POLICY", cust001_context)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What documents are required for KYC?"}]}, config
    )

    assert result["current_agent"] == "policy_agent"
    assert "PAN card" in result["final_answer"]
    nodes = _node_ids(result["execution_events"])
    assert "enterprise_rag" in nodes
    assert "banking_agent" not in nodes
    assert "dispute_agent" not in nodes


def test_banking_query_routes_through_banking_agent_and_scoped_tool(
    graph_db_path, fake_pipeline, cust001_context, fake_nemo
):
    llm = StubLLMClient(['{"route": "BANKING", "confidence": 0.9, "reason": "own transactions"}'])
    graph = build_graph(
        llm=llm, enterprise_pipeline=fake_pipeline, db_path=graph_db_path, nemo=fake_nemo
    )
    config = run_config("SESS-BANKING", cust001_context)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Show my recent transactions."}]}, config
    )

    assert result["current_agent"] == "banking_agent"
    assert "TX1007" in result["final_answer"] or "QuickPay" in result["final_answer"]
    event_labels = [e.label for e in result["execution_events"]]
    assert "get_recent_transactions" in event_labels


def test_dispute_workflow_multiturn_then_approve_creates_dispute(
    graph_db_path, fake_pipeline, cust001_context, fake_nemo
):
    supervisor_reply = '{"route": "DISPUTE", "confidence": 0.9, "reason": "unrecognized txn"}'
    llm = StubLLMClient([supervisor_reply, supervisor_reply])
    graph = build_graph(
        llm=llm, enterprise_pipeline=fake_pipeline, db_path=graph_db_path, nemo=fake_nemo
    )
    config = run_config("SESS-DISPUTE-1", cust001_context)

    turn1 = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "I don't recognize one of my recent transactions."}
            ]
        },
        config,
    )
    assert "QuickPay" in turn1["final_answer"]
    assert not is_waiting_for_approval(graph, config)

    turn2 = graph.invoke({"messages": [{"role": "user", "content": "The ₹4,500 one."}]}, config)
    assert "__interrupt__" in turn2
    assert is_waiting_for_approval(graph, config)
    state = graph.get_state(config)
    assert state.values["selected_transaction_id"] == "TX1007"
    assert state.values["approval_required"] is True

    resumed = resume_graph(graph, config, approved=True)
    assert "Dispute created" in resumed["final_answer"]
    assert not is_waiting_for_approval(graph, config)

    with banking_data.session(graph_db_path) as conn:
        row = conn.execute("SELECT * FROM disputes WHERE transaction_id = 'TX1007'").fetchone()
    assert row is not None


def test_dispute_workflow_rejection_never_creates_dispute(
    graph_db_path, fake_pipeline, cust001_context, fake_nemo
):
    supervisor_reply = '{"route": "DISPUTE", "confidence": 0.9, "reason": "unrecognized txn"}'
    llm = StubLLMClient([supervisor_reply, supervisor_reply])
    graph = build_graph(
        llm=llm, enterprise_pipeline=fake_pipeline, db_path=graph_db_path, nemo=fake_nemo
    )
    config = run_config("SESS-DISPUTE-REJECT", cust001_context)

    graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "I don't recognize one of my recent transactions."}
            ]
        },
        config,
    )
    graph.invoke({"messages": [{"role": "user", "content": "The ₹4,500 one."}]}, config)
    assert is_waiting_for_approval(graph, config)

    resumed = resume_graph(graph, config, approved=False)
    assert "cancelled" in resumed["final_answer"].lower()

    tool_events = [e for e in resumed["execution_events"] if e.node_id == "create_dispute"]
    assert tool_events == []

    with banking_data.session(graph_db_path) as conn:
        row = conn.execute("SELECT * FROM disputes WHERE transaction_id = 'TX1007'").fetchone()
    assert row is None


def test_customer_only_ever_sees_own_transactions_in_dispute_flow(
    graph_db_path, fake_pipeline, cust002_context, fake_nemo
):
    llm = StubLLMClient(['{"route": "DISPUTE", "confidence": 0.9, "reason": "unrecognized txn"}'])
    graph = build_graph(
        llm=llm, enterprise_pipeline=fake_pipeline, db_path=graph_db_path, nemo=fake_nemo
    )
    config = run_config("SESS-CUST2", cust002_context)

    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "I don't recognize one of my recent transactions."}
            ]
        },
        config,
    )
    assert "BigBasket" in result["final_answer"]
    assert "QuickPay" not in result["final_answer"]
    assert "TX1007" not in result["final_answer"]
