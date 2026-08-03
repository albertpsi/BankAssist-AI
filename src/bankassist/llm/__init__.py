"""LLM access — the single provider chokepoint (ADR-0005).

Every model call in the application goes through ``LLMClient``. That one seam is
what lets the provider be swapped by configuration, lets token accounting happen
in one place, and lets every test run without a network or an API key.

Scope note: this package knows how to send messages and return a typed response.
It knows nothing about retrieval, agents, guardrails, caching, or evaluation —
those compose *on top of* it in later labs.
"""

from bankassist.llm.base import LLMClient, LLMMessage, LLMResponse, TokenUsage
from bankassist.llm.factory import build_llm_client
from bankassist.llm.openai_client import OpenAIClient
from bankassist.llm.stub import StubLLMClient

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "OpenAIClient",
    "StubLLMClient",
    "TokenUsage",
    "build_llm_client",
]
