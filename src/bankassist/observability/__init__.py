"""AgentOps operational observability (Lab 6, ADR-0012).

This package is the sole integration point with the AgentOps SDK. Nothing
outside it imports ``agentops`` directly, so the dependency stays swappable
and every test can run with it fully disabled (ADR-0012, CLAUDE.md §21).

Responsibility boundary (see docs/requirements/lab-06-agentops-evaluation.md
§3): ``ExecutionEvent`` remains the BankAssist-facing "what happened" model
rendered in the Streamlit UI; this package answers "how did the agentic
system execute" for the AgentOps dashboard. Neither replaces the other.
"""

from __future__ import annotations

from bankassist.observability.agentops_client import init_agentops, is_enabled, shutdown_agentops
from bankassist.observability.decorators import (
    agent_span,
    operation,
    run,
    tool_span,
    trace,
    update_metadata,
    workflow,
)

__all__ = [
    "agent_span",
    "init_agentops",
    "is_enabled",
    "operation",
    "run",
    "shutdown_agentops",
    "tool_span",
    "trace",
    "update_metadata",
    "workflow",
]
