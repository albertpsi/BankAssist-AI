"""The ``Classifier`` seam (FR-L3-3)."""

from __future__ import annotations

from typing import Protocol

from bankassist.rag.models.classification_result import ClassificationResult


class Classifier(Protocol):
    def execute(self, question: str) -> ClassificationResult:
        """Classify a question into one of the seven defined routes."""
        ...
