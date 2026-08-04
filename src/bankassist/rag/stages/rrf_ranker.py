"""Manual Reciprocal Rank Fusion (FR-L3-7). No external RRF library.

``score = sum(1 / (k + rank))`` over every list a chunk appears in, rank
1-indexed within that list. A chunk in both lists accumulates both terms.
"""

from __future__ import annotations

from bankassist.logging_config import get_logger
from bankassist.rag.models.retrieval_result import (
    HybridRetrievalResult,
    RetrievalResult,
    RRFEntry,
    RRFResult,
    ScoredChunk,
)

logger = get_logger(__name__)

_Key = tuple[str, int]


class RRFRanker:
    def __init__(self, k: int = 60) -> None:
        self._k = k

    def execute(self, filtered: HybridRetrievalResult) -> RRFResult:
        entries: dict[_Key, RRFEntry] = {}
        self._absorb_vector(entries, filtered.vector)
        self._absorb_bm25(entries, filtered.bm25)

        ordered = sorted(entries.values(), key=lambda entry: entry.rrf_score, reverse=True)

        logger.info(
            "rrf fusion",
            extra={
                "k": self._k,
                "fused_count": len(ordered),
                "ranking": [
                    {
                        "document": e.metadata.document,
                        "chunk_index": e.chunk_index,
                        "score": e.rrf_score,
                    }
                    for e in ordered
                ],
            },
        )
        return RRFResult(entries=ordered)

    def _absorb_vector(self, entries: dict[_Key, RRFEntry], result: RetrievalResult) -> None:
        for rank, chunk in enumerate(result.results, start=1):
            key = _key(chunk)
            contribution = 1.0 / (self._k + rank)
            entries[key] = RRFEntry(
                text=chunk.text,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                vector_rank=rank,
                vector_score=chunk.score,
                rrf_score=contribution,
            )

    def _absorb_bm25(self, entries: dict[_Key, RRFEntry], result: RetrievalResult) -> None:
        for rank, chunk in enumerate(result.results, start=1):
            key = _key(chunk)
            contribution = 1.0 / (self._k + rank)
            existing = entries.get(key)
            if existing is None:
                entries[key] = RRFEntry(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    chunk_index=chunk.chunk_index,
                    bm25_rank=rank,
                    bm25_score=chunk.score,
                    rrf_score=contribution,
                )
            else:
                entries[key] = existing.model_copy(
                    update={
                        "bm25_rank": rank,
                        "bm25_score": chunk.score,
                        "rrf_score": existing.rrf_score + contribution,
                    }
                )


def _key(chunk: ScoredChunk) -> _Key:
    return (chunk.metadata.document, chunk.chunk_index)
