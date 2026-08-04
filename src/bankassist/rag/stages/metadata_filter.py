"""Optional metadata narrowing, applied before fusion (FR-L3-6)."""

from __future__ import annotations

from bankassist.logging_config import get_logger
from bankassist.rag.models.retrieval_context import MetadataFilters
from bankassist.rag.models.retrieval_result import (
    HybridRetrievalResult,
    RetrievalResult,
    ScoredChunk,
)

logger = get_logger(__name__)


class MetadataFilter:
    """Exact-match filtering on ``category``/``document``/``source``, ANDed.

    ``filters=None`` (or an empty ``MetadataFilters``) is a no-op passthrough
    (FR-L3-6.2) — filtering is opt-in.
    """

    def execute(
        self, results: HybridRetrievalResult, filters: MetadataFilters | None
    ) -> HybridRetrievalResult:
        if filters is None or filters.is_empty():
            return results

        filtered = HybridRetrievalResult(
            vector=_filter_result(results.vector, filters),
            bm25=_filter_result(results.bm25, filters),
        )
        logger.info(
            "metadata filter",
            extra={
                "filters": filters.model_dump(exclude_none=True),
                "vector_before": len(results.vector.results),
                "vector_after": len(filtered.vector.results),
                "bm25_before": len(results.bm25.results),
                "bm25_after": len(filtered.bm25.results),
            },
        )
        return filtered


def _filter_result(result: RetrievalResult, filters: MetadataFilters) -> RetrievalResult:
    kept = [chunk for chunk in result.results if _matches(chunk, filters)]
    return result.model_copy(update={"results": kept})


def _matches(chunk: ScoredChunk, filters: MetadataFilters) -> bool:
    if filters.category is not None and chunk.metadata.category != filters.category:
        return False
    if filters.document is not None and chunk.metadata.document != filters.document:
        return False
    return not (filters.source is not None and chunk.metadata.source != filters.source)
