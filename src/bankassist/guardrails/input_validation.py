"""Deterministic input validation (Lab 5) — runs before any LLM call.

Cheap, decidable checks that reject malformed requests without spending a NeMo
Guardrails LLM call on input that can already be rejected deterministically. This is
intentional ordering (see ADR-0011): deterministic controls execute before
probabilistic ones whenever they can reject invalid input more cheaply and reliably.

``AgentChatRequest`` already enforces min/max length and blank-rejection at the
Pydantic boundary (``api/schemas.py``); this module re-asserts the same properties as
an explicit, traceable guardrail step (so it emits its own ``ExecutionEvent``) and is
also usable anywhere a raw string needs validating without going through the API layer.
"""

from __future__ import annotations

from bankassist.guardrails.models import GuardrailCategory, GuardrailResult

MAX_MESSAGE_CHARS = 2000

GUARDRAIL_ID = "input_validation.shape"


def validate_message_shape(message: str, *, max_chars: int = MAX_MESSAGE_CHARS) -> GuardrailResult:
    """Reject empty, whitespace-only, or oversized input.

    Deliberately narrow: shape only, never semantics. Semantic/intent checks
    (injection, jailbreak) belong to the NeMo input rail that runs after this.
    """
    if not message or not message.strip():
        return GuardrailResult.block(
            GUARDRAIL_ID,
            GuardrailCategory.INPUT,
            reason="Your message appears to be empty.",
            internal_reason="blank_input",
        )
    if len(message) > max_chars:
        return GuardrailResult.block(
            GUARDRAIL_ID,
            GuardrailCategory.INPUT,
            reason="Your message is too long.",
            internal_reason=f"oversized_input:{len(message)}>{max_chars}",
            metadata={"length": len(message), "max_chars": max_chars},
        )
    return GuardrailResult.allow(GUARDRAIL_ID, GuardrailCategory.INPUT, reason="Input shape valid.")
