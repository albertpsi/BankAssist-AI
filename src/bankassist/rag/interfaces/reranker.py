"""The ``Reranker`` seam (FR-L3-8, ADR-0008)."""

from __future__ import annotations

from typing import Protocol

from bankassist.rag.models.rerank_result import RerankResult
from bankassist.rag.models.retrieval_result import RRFResult


class Reranker(Protocol):
    def execute(self, query: str, fused: RRFResult, top_n: int = 5) -> RerankResult:
        """Re-score the fused candidates against ``query`` and return the top ``top_n``."""
        ...
