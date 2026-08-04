"""Query classification (FR-L3-3)."""

from __future__ import annotations

import json
import time

from bankassist.llm.base import LLMClient, LLMMessage
from bankassist.logging_config import get_logger
from bankassist.rag.models.classification_result import (
    ClassificationLabel,
    ClassificationResult,
)

logger = get_logger(__name__)

LABELS: tuple[ClassificationLabel, ...] = (
    "Policy",
    "FAQ",
    "Procedure",
    "Eligibility",
    "Definition",
    "Comparison",
    "Unknown",
)

_SYSTEM_PROMPT = f"""You classify a banking-policy question into exactly one label.

Labels: {", ".join(LABELS[:-1])}.

Reply with ONLY a JSON object of the shape:
{{"label": "<one of the labels above>", "confidence": <number between 0 and 1>}}"""


class QueryClassifier:
    """Calls the configured classifier model and returns a structured label.

    A malformed or unparseable response classifies as ``Unknown`` with
    confidence 0.0 rather than raising (FR-L3-3.3) — classification informs the
    pipeline, it does not gate it.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def execute(self, question: str) -> ClassificationResult:
        started = time.perf_counter()
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=question),
        ]

        result = self._classify(messages)
        result = result.model_copy(
            update={"latency_ms": round((time.perf_counter() - started) * 1000.0, 2)}
        )

        logger.info(
            "query classification",
            extra={
                "original_question": question,
                "label": result.label,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            },
        )
        return result

    def _classify(self, messages: list[LLMMessage]) -> ClassificationResult:
        try:
            response = self._llm.complete(messages, tier="classifier")
            payload = json.loads(response.text)
            label = payload["label"]
            confidence = float(payload["confidence"])
        except Exception:
            return ClassificationResult(label="Unknown", confidence=0.0)

        if label not in LABELS:
            return ClassificationResult(label="Unknown", confidence=0.0)
        return ClassificationResult(label=label, confidence=max(0.0, min(1.0, confidence)))
