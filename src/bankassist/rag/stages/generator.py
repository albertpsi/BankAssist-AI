"""Grounded generation (FR-L3-10). Delegates to the existing ``LLMClient``."""

from __future__ import annotations

import time

from pydantic import BaseModel

from bankassist.llm.base import LLMClient
from bankassist.logging_config import get_logger
from bankassist.rag.models.retrieval_context import PromptContext
from bankassist.rag.prompts import REFUSAL

logger = get_logger(__name__)


class GenerationResult(BaseModel):
    answer: str
    grounded: bool
    latency_ms: float = 0.0


class Generator:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def execute(self, context: PromptContext) -> GenerationResult:
        # FR-L3-10.2: nothing survived retrieval/filter/rerank — refuse without
        # calling the model, same contract as `BasicRagPipeline.answer()`.
        if context.chunk_count == 0:
            return GenerationResult(answer=REFUSAL, grounded=False)

        started = time.perf_counter()
        response = self._llm.complete(context.messages)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        text = response.text.strip()
        grounded = text != REFUSAL

        logger.info("generation complete", extra={"grounded": grounded, "latency_ms": latency_ms})
        return GenerationResult(answer=text, grounded=grounded, latency_ms=latency_ms)
