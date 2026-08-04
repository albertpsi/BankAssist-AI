"""Shared test double for ``NemoGuardrailsAdapter`` (Lab 5).

Duck-types the adapter without any NeMo/LLM call, so every test using it is
deterministic and never requires a live OpenAI call (Lab 5 §5/§6 requirement).
Defaults to ALLOW for both rails; flip ``block_input``/``block_output`` to exercise
the BLOCK path.
"""

from __future__ import annotations

from bankassist.guardrails.models import GuardrailCategory, GuardrailResult


class FakeNemoAdapter:
    def __init__(self) -> None:
        self.block_input = False
        self.block_output = False
        self.input_calls: list[str] = []
        self.output_calls: list[str] = []

    def check_input(self, user_input: str) -> GuardrailResult:
        self.input_calls.append(user_input)
        if self.block_input:
            return GuardrailResult.block(
                "nemo.input_rail.self_check",
                GuardrailCategory.INPUT,
                reason="Your message was flagged by our safety check and can't be processed.",
                internal_reason="fake_nemo_block_input",
            )
        return GuardrailResult.allow(
            "nemo.input_rail.self_check", GuardrailCategory.INPUT, reason="NeMo input rail passed."
        )

    def check_output(self, bot_response: str) -> GuardrailResult:
        self.output_calls.append(bot_response)
        if self.block_output:
            return GuardrailResult.block(
                "nemo.output_rail.self_check",
                GuardrailCategory.OUTPUT,
                reason="This response could not be delivered as generated.",
                internal_reason="fake_nemo_block_output",
            )
        return GuardrailResult.allow(
            "nemo.output_rail.self_check",
            GuardrailCategory.OUTPUT,
            reason="NeMo output rail passed.",
        )
