"""NeMo Guardrails adapter — the only module in this codebase that imports NeMo.

Everything downstream sees :class:`~bankassist.guardrails.models.GuardrailResult`,
never a NeMo type (ADR-0011: NeMo is replaceable without touching the rest of the
app). Two rails only, both built from NeMo's built-in "self check input" flow with a
BankAssist-specific prompt — no hand-written Colang dialogue system. The output rail
reuses the same mechanism as a second, independent config rather than NeMo's "self
check output" flow; see the comment in ``nemo_config/output_rail.yml`` for why.

Production uses the real ``ChatOpenAI`` model, pointed at the same OpenAI credential
and the same economical tier (``Settings.llm_model_fast``) the rest of the app already
uses — no second LLM provider or API key. Tests inject a fake LangChain LLM via
``llm_override`` so guardrail behaviour is fully deterministic and never requires a
live OpenAI call (Lab 5 §5/§6 requirement).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter

from bankassist.config import Settings
from bankassist.guardrails.models import GuardrailCategory, GuardrailResult
from bankassist.observability import run as observability_run

_CONFIG_DIR = Path(__file__).parent / "nemo_config"

INPUT_GUARDRAIL_ID = "nemo.input_rail.self_check"
OUTPUT_GUARDRAIL_ID = "nemo.output_rail.self_check"

_REFUSAL_MARKERS = ("I'm sorry, I can't respond to that", "I can't respond to that")


def _load_config(filename: str) -> RailsConfig:
    yaml_content = (_CONFIG_DIR / filename).read_text(encoding="utf-8")
    return RailsConfig.from_content(yaml_content=yaml_content)


def _build_llm(settings: Settings) -> Any:
    """The real model NeMo calls in production — same credential, same tier."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model_fast,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _triggered(response: Any) -> bool:
    content = response.get("content", "") if isinstance(response, dict) else str(response)
    return any(marker in content for marker in _REFUSAL_MARKERS)


class NemoGuardrailsAdapter:
    """Wraps two ``LLMRails`` instances — one per rail direction.

    ``llm_override`` exists solely for tests — a LangChain fake LLM whose canned
    responses drive a deterministic ALLOW/BLOCK verdict without a network call. When
    provided, it is used for *both* rails (tests construct one adapter per scenario).
    """

    def __init__(self, settings: Settings, *, llm_override: Any | None = None) -> None:
        raw_llm = llm_override if llm_override is not None else _build_llm(settings)
        llm = LangChainLLMAdapter(raw_llm)
        self._input_rails = LLMRails(_load_config("input_rail.yml"), llm=llm)
        self._output_rails = LLMRails(_load_config("output_rail.yml"), llm=llm)

    def check_input(self, user_input: str) -> GuardrailResult:
        # Named AgentOps span (Lab 6 requirements §5): a guardrail verdict is
        # neither an LLM call nor a tool call from AgentOps' point of view,
        # even though it calls an LLM internally (which AgentOps does capture
        # automatically) — this labels the verdict boundary itself.
        response = observability_run(
            "operation",
            "guardrail.input_rail",
            self._input_rails.generate,
            messages=[{"role": "user", "content": user_input}],
        )
        if _triggered(response):
            return GuardrailResult.block(
                INPUT_GUARDRAIL_ID,
                GuardrailCategory.INPUT,
                reason="Your message was flagged by our safety check and can't be processed.",
                internal_reason="nemo_self_check_input_triggered",
            )
        return GuardrailResult.allow(
            INPUT_GUARDRAIL_ID, GuardrailCategory.INPUT, reason="NeMo input rail passed."
        )

    def check_output(self, bot_response: str) -> GuardrailResult:
        """Classifies an already-generated agent answer. Never rewrites it — deterministic

        redaction/masking (``guardrails.redaction``/``guardrails.masking``) is the layer
        responsible for altering output text; this only classifies it.
        """
        response = observability_run(
            "operation",
            "guardrail.output_rail",
            self._output_rails.generate,
            messages=[{"role": "user", "content": bot_response}],
        )
        if _triggered(response):
            return GuardrailResult.block(
                OUTPUT_GUARDRAIL_ID,
                GuardrailCategory.OUTPUT,
                reason="This response could not be delivered as generated.",
                internal_reason="nemo_self_check_output_triggered",
            )
        return GuardrailResult.allow(
            OUTPUT_GUARDRAIL_ID, GuardrailCategory.OUTPUT, reason="NeMo output rail passed."
        )
