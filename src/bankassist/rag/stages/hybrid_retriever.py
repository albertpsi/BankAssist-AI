"""Composes the vector and BM25 result sets (FR-L3-5.3).

A thin, pure composition stage — it does not rank or filter, only pairs the two
independent result sets for the stages that do.
"""

from __future__ import annotations

from bankassist.logging_config import get_logger
from bankassist.rag.models.retrieval_result import HybridRetrievalResult, RetrievalResult

logger = get_logger(__name__)


class HybridRetriever:
    def execute(self, vector: RetrievalResult, bm25: RetrievalResult) -> HybridRetrievalResult:
        result = HybridRetrievalResult(vector=vector, bm25=bm25)

        vector_docs = {r.metadata.document for r in vector.results}
        bm25_docs = {r.metadata.document for r in bm25.results}
        logger.info(
            "hybrid retrieval",
            extra={
                "vector_count": len(vector.results),
                "bm25_count": len(bm25.results),
                "overlap_count": len(vector_docs & bm25_docs),
                "vector_only_count": len(vector_docs - bm25_docs),
                "bm25_only_count": len(bm25_docs - vector_docs),
            },
        )
        return result
