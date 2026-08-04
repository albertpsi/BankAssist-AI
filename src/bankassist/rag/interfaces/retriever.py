"""The ``Retriever`` seam shared by the vector and BM25 legs (FR-L3-5)."""

from __future__ import annotations

from typing import Protocol

from bankassist.rag.models.retrieval_result import RetrievalResult


class Retriever(Protocol):
    def execute(self, query: str, top_k: int = 20) -> RetrievalResult:
        """Return this retriever's independent top-``top_k`` result set."""
        ...
