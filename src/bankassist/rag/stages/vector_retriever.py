"""Dense (vector) retrieval leg of hybrid search (FR-L3-5.1)."""

from __future__ import annotations

import time

from bankassist.logging_config import get_logger
from bankassist.rag.embeddings import Embedder
from bankassist.rag.models.retrieval_result import RetrievalResult, ScoredChunk
from bankassist.rag.vector_store import VectorStore

logger = get_logger(__name__)


class VectorRetriever:
    """Reuses the existing ``Embedder``/``VectorStore`` — no new retrieval code."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def execute(self, query: str, top_k: int = 20) -> RetrievalResult:
        started = time.perf_counter()
        vector = self._embedder.embed_query(query)
        chunks = self._store.query(vector, top_k=top_k)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        results = [
            ScoredChunk(
                text=chunk.text,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
            )
            for chunk in chunks
        ]

        logger.info(
            "vector retrieval",
            extra={
                "query": query,
                "top_k": top_k,
                "result_count": len(results),
                "documents": [r.metadata.document for r in results],
                "scores": [round(r.score, 6) for r in results],
                "latency_ms": latency_ms,
            },
        )
        return RetrievalResult(query=query, results=results, latency_ms=latency_ms)
