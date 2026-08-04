from bankassist.errors import AuthorizationError
from bankassist.execution_event import ExecutionEventType, ExecutionStatus
from bankassist.tools.dispatcher import call_tool


def test_call_tool_emits_started_and_completed_on_success():
    result, events = call_tool(node_id="n1", label="a_tool", fn=lambda: "ok")
    assert result == "ok"
    assert [e.event_type for e in events] == [
        ExecutionEventType.TOOL_STARTED,
        ExecutionEventType.TOOL_COMPLETED,
    ]
    assert events[-1].status == ExecutionStatus.COMPLETED


def test_call_tool_emits_failed_status_on_authorization_error_without_leaking_details():
    def boom():
        raise AuthorizationError("denied", details={"permission": "x"})

    result, events = call_tool(node_id="n1", label="a_tool", fn=boom)
    assert result is None
    assert events[-1].status == ExecutionStatus.FAILED
    assert "denied" in events[-1].summary
