"""Query rewriting for retrieval (FR-L3-4)."""

from __future__ import annotations

import time

from bankassist.llm.base import LLMClient, LLMMessage
from bankassist.logging_config import get_logger
from bankassist.rag.models.classification_result import ClassificationResult
from bankassist.rag.models.rewrite_result import RewriteResult

logger = get_logger(__name__)

_SYSTEM_PROMPT = """Rewrite the user's banking question into a clear, retrieval-oriented
query: resolve implicit referents, expand banking abbreviations (APR, ATM, ACH, POS, KYC),
and state the topic explicitly. Reply with ONLY the rewritten question, nothing else."""


class QueryRewriter:
    """Rewrites a question for retrieval, never discarding the original.

    A failed or empty rewrite falls back to the original question unchanged
    (FR-L3-4.3) — a rewrite failure is never a pipeline failure.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def execute(self, question: str, classification: ClassificationResult) -> RewriteResult:
        started = time.perf_counter()
        rewritten, fallback = self._rewrite(question, classification)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        result = RewriteResult(
            original_question=question,
            rewritten_question=rewritten,
            latency_ms=latency_ms,
            fallback_used=fallback,
        )
        logger.info(
            "query rewrite",
            extra={
                "original_question": result.original_question,
                "rewritten_question": result.rewritten_question,
                "fallback_used": fallback,
                "latency_ms": latency_ms,
            },
        )
        return result

    def _rewrite(
        self, question: str, classification: ClassificationResult
    ) -> tuple[str, bool]:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"Classified as: {classification.label}\nQuestion: {question}",
            ),
        ]
        try:
            response = self._llm.complete(messages, tier="fast")
            rewritten = response.text.strip()
        except Exception:
            return question, True

        if not rewritten:
            return question, True
        return rewritten, False
