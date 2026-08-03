"""OpenAI adapter.

The only module in the application that imports the provider SDK. Provider
exceptions are wrapped in ``LLMError`` here so they never leak past this package.
"""

from __future__ import annotations

import time

from openai import OpenAI, OpenAIError

from bankassist.config import ModelTier, Settings
from bankassist.errors import ConfigurationError, LLMError
from bankassist.llm.base import LLMMessage, LLMResponse, TokenUsage
from bankassist.logging_config import get_logger
from bankassist.tracing.span import SpanType
from bankassist.tracing.tracer import NoOpTracer, Tracer

logger = get_logger(__name__)


class OpenAIClient:
    """Implements ``LLMClient`` against the OpenAI Chat Completions API."""

    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        # Reachable: Settings only enforces the key when llm_provider == "openai",
        # so a caller can construct this directly with a differently-configured
        # Settings. Fail here rather than let the SDK do it later.
        if settings.openai_api_key is None:
            raise ConfigurationError(
                "OpenAIClient requires OPENAI_API_KEY to be configured.",
                details={"field": "openai_api_key"},
            )

        self._settings = settings
        self._tracer = tracer or NoOpTracer()
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        tier: ModelTier = "fast",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = self._settings.model_for_tier(tier)
        # `is None` rather than `or`: an explicit 0 is a caller error worth
        # surfacing, not something to silently replace with the default.
        limit = self._settings.llm_max_tokens if max_tokens is None else max_tokens

        span_ctx = self._tracer.span(SpanType.LLM_CALL, "openai.complete", model=model, tier=tier)
        with span_ctx as span:
            started = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=[m.model_dump() for m in messages],
                    max_completion_tokens=limit,
                )
            except OpenAIError as exc:
                # `from exc` keeps the provider traceback for the logs; the message
                # we surface names the type only, never the request or the key.
                raise LLMError(
                    f"OpenAI request failed: {type(exc).__name__}",
                    details={"model": model, "provider": "openai"},
                ) from exc

            latency_ms = (time.perf_counter() - started) * 1000.0
            response = self._to_response(completion, model, latency_ms)

            span.set_attribute("input_tokens", response.usage.input_tokens)
            span.set_attribute("output_tokens", response.usage.output_tokens)
            span.set_attribute("finish_reason", response.finish_reason)

        logger.info(
            "llm call complete",
            extra={
                "model": model,
                "tier": tier,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response

    @staticmethod
    def _to_response(completion: object, model: str, latency_ms: float) -> LLMResponse:
        """Map an SDK completion onto our own type.

        Read defensively: the SDK's optional fields differ between models, and a
        missing usage block should not crash a working call.
        """
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError(
                "OpenAI returned no choices.", details={"model": model, "provider": "openai"}
            )

        message = getattr(choices[0], "message", None)
        usage = getattr(completion, "usage", None)

        return LLMResponse(
            text=getattr(message, "content", None) or "",
            model=getattr(completion, "model", model),
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            finish_reason=getattr(choices[0], "finish_reason", None),
        )
