"""Lab 5 guardrail layer.

Two kinds of control live here, and the split is the point (see
``docs/decisions/0011-nemo-guardrails-for-ai-semantic-rails.md``):

- **AI-semantic rails** (``nemo_adapter``) — fuzzy, LLM-judged properties: prompt
  injection/jailbreak intent, system-prompt extraction attempts, and semantic output
  safety. Backed by NeMo Guardrails.
- **Deterministic application security** (``input_validation``, ``tool_authorization``,
  ``masking``, ``redaction``) — decidable properties that must hold even if the LLM
  guardrail is unavailable, wrong, or bypassed. These never depend on a model call.

Every check in this package returns a :class:`~bankassist.guardrails.models.GuardrailResult`
so the caller can build an :class:`~bankassist.execution_event.ExecutionEvent` from it —
this package never talks to LangGraph state or the API layer directly.
"""

from __future__ import annotations

from bankassist.guardrails.models import GuardrailAction, GuardrailCategory, GuardrailResult

__all__ = ["GuardrailAction", "GuardrailCategory", "GuardrailResult"]
