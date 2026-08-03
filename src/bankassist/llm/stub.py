"""A scripted LLM client for tests.

Every later lab depends on this: it is what lets the whole suite run with no API
key, no network, and no cost, while still asserting on what was sent to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from bankassist.config import ModelTier
from bankassist.errors import LLMError
from bankassist.llm.base import LLMMessage, LLMResponse, TokenUsage


@dataclass(frozen=True)
class RecordedCall:
    """What a caller asked the model to do."""

    messages: list[LLMMessage]
    tier: ModelTier
    max_tokens: int | None


class StubLLMClient:
    """Returns queued responses in order and records the calls it received.

    Assert on routing and prompt structure, never on model prose — see the
    ``testing`` skill.
    """

    def __init__(self, responses: list[LLMResponse | str] | None = None) -> None:
        self._queue: list[LLMResponse | str] = list(responses or [])
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        tier: ModelTier = "fast",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(RecordedCall(messages=messages, tier=tier, max_tokens=max_tokens))

        if not self._queue:
            raise LLMError(
                "StubLLMClient script is exhausted: the code under test made more "
                f"calls than the {len(self.calls) - 1} response(s) provided.",
                details={"calls_made": len(self.calls)},
            )

        queued = self._queue.pop(0)
        if isinstance(queued, LLMResponse):
            return queued
        return LLMResponse(
            text=queued,
            model=f"stub-{tier}",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            finish_reason="stop",
        )

    def last_call(self) -> RecordedCall:
        """Return the most recent call, for assertions."""
        if not self.calls:
            raise AssertionError("StubLLMClient received no calls")
        return self.calls[-1]
