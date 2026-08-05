"""Sanitize custom span/trace attributes before they leave the process.

AgentOps has no documented built-in redaction/masking (verified against the
current SDK docs — see docs/decisions/0012-agentops-observability.md). Its
automatic OpenAI instrumentation is AgentOps' own concern; this module only
guards the *custom* attributes BankAssist code attaches via
``observability.decorators`` — route names, tool names, guardrail verdicts,
latencies. It reuses the existing deterministic patterns
(``guardrails.redaction.redact``, ``guardrails.masking.mask_sensitive_identifiers``)
rather than duplicating them, and additionally drops any attribute whose key
looks like a credential outright (CLAUDE.md §6, Lab 6 requirements §8).
"""

from __future__ import annotations

import re
from typing import Any

from bankassist.guardrails.masking import mask_sensitive_identifiers
from bankassist.guardrails.redaction import redact

# Whole-word markers, not substrings: a raw substring check would also catch
# and drop a legitimate attribute like `tokens`/`output_tokens` (LLM token
# counts) just because it contains "token". Keys are split on non-alphanumeric
# characters before comparing, so `access_token` and `apikey` both match but
# `input_tokens` does not.
_FORBIDDEN_KEY_WORDS = {
    "authorization",
    "jwt",
    "token",
    "apikey",
    "password",
    "secret",
    "systemprompt",
}


def _key_is_forbidden(key: str) -> bool:
    words = re.split(r"[^a-z0-9]+", key.lower())
    combined = "".join(words)
    return bool(set(words) & _FORBIDDEN_KEY_WORDS) or combined in _FORBIDDEN_KEY_WORDS


def sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop credential-shaped keys; redact/mask secret- and PII-shaped values.

    Never raises — an attribute this cannot safely characterize is dropped
    rather than sent, since AgentOps observability is not the record of
    truth and a missing attribute is a much smaller cost than a leaked one.
    """
    clean: dict[str, Any] = {}
    for key, value in attributes.items():
        if _key_is_forbidden(key):
            continue
        if isinstance(value, str):
            clean[key] = mask_sensitive_identifiers(redact(value))
        elif isinstance(value, int | float | bool) or value is None:
            clean[key] = value
        else:
            # Anything else (dict, list, model) is not a known-safe shape for
            # a trace attribute — stringify only its type, never its content.
            clean[key] = f"<{type(value).__name__}>"
    return clean
