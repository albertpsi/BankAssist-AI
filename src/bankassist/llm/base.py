"""The LLM client protocol and its message/response types."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from bankassist.config import ModelTier

Role = Literal["system", "user", "assistant"]


class LLMMessage(BaseModel):
    """One turn of a conversation sent to the model."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token counts reported by the provider for a single call."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """A completed model call, with the accounting Labs 6 and 7 read."""

    text: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    finish_reason: str | None = None


class LLMClient(Protocol):
    """Sends messages to a model and returns a typed response.

    Deliberately narrow. Structured outputs, tool calling, and streaming are added
    by the labs that need them, not anticipated here.
    """

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        tier: ModelTier = "fast",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a completion.

        Args:
            messages: the conversation, in order.
            tier: which configured model tier to use. Callers ask for a tier, never
                a model id — that keeps model selection in configuration.
            max_tokens: overrides the configured default when set.

        Raises:
            LLMError: the provider call failed.
        """
        ...
