"""Supervisor Agent (Lab 4, §2). Routing only — no retrieval, no tools, no DB access."""

from __future__ import annotations

import json

from pydantic import BaseModel

from bankassist.llm.base import LLMClient, LLMMessage

VALID_ROUTES = ("POLICY", "BANKING", "DISPUTE", "CLARIFICATION", "UNSUPPORTED")

_SYSTEM_PROMPT = """You are the routing supervisor for a banking assistant. Classify the
user's most recent message into exactly one route:

- POLICY: questions about banking/card policy, KYC, eligibility, procedures, FAQs.
- BANKING: requests about the caller's own accounts, balances, or transaction history.
- DISPUTE: unrecognized transactions, transaction disputes, chargeback questions.
- CLARIFICATION: the request is ambiguous and needs a follow-up question.
- UNSUPPORTED: outside banking (e.g. investment advice, unrelated topics).

Reply with ONLY a JSON object:
{"route": "<one of the routes above>", "confidence": <0..1>,
 "reason": "<short operational reason, no chain-of-thought>"}"""


class SupervisorDecision(BaseModel):
    route: str
    confidence: float
    reason: str


def decide_route(
    llm: LLMClient, message: str, history: list[str] | None = None
) -> SupervisorDecision:
    """Ask the model for a structured routing decision.

    A malformed/unparseable response routes to CLARIFICATION with confidence 0 rather
    than raising — routing informs the graph, it does not crash it, matching the
    pattern already used by Lab 3's ``QueryClassifier``.
    """
    context = "\n".join(history or [])
    user_content = f"{context}\n\nLatest message: {message}" if context else message

    messages = [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]
    response = llm.complete(messages, tier="classifier")

    try:
        payload = json.loads(response.text)
        route = str(payload["route"]).upper()
        confidence = max(0.0, min(1.0, float(payload["confidence"])))
        reason = str(payload["reason"])[:280]
    except Exception:
        return SupervisorDecision(
            route="CLARIFICATION", confidence=0.0, reason="Could not parse routing decision."
        )

    if route not in VALID_ROUTES:
        return SupervisorDecision(
            route="CLARIFICATION", confidence=0.0, reason="Model returned an unknown route."
        )
    return SupervisorDecision(route=route, confidence=confidence, reason=reason)
