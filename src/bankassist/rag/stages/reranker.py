"""Cross-encoder reranking (FR-L3-8, ADR-0008).

``cross-encoder/ms-marco-MiniLM-L-6-v2`` loads once per process — construction
is expensive (model download + weight load), so callers should build one
``CrossEncoderReranker`` and reuse it, the same lazy-singleton pattern
``api/routes/rag.py::get_pipeline`` already uses for the Pinecone client.
"""

from __future__ import annotations

import time

from bankassist.logging_config import get_logger
from bankassist.rag.models.rerank_result import RerankEntry, RerankResult
from bankassist.rag.models.retrieval_result import RRFResult

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def execute(self, query: str, fused: RRFResult, top_n: int = 5) -> RerankResult:
        started = time.perf_counter()
        candidates = fused.entries

        before = [
            {
                "document": e.metadata.document,
                "chunk_index": e.chunk_index,
                "rrf_score": e.rrf_score,
            }
            for e in candidates
        ]

        if not candidates:
            logger.info("reranking", extra={"before": before, "after": [], "latency_ms": 0.0})
            return RerankResult(entries=[])

        pairs = [(query, entry.text) for entry in candidates]
        scores = self._model.predict(pairs)

        pre_ranked = list(enumerate(candidates, start=1))
        post_ranked = sorted(
            zip(pre_ranked, scores, strict=True), key=lambda item: item[1], reverse=True
        )[:top_n]

        entries = [
            RerankEntry(
                text=entry.text,
                metadata=entry.metadata,
                chunk_index=entry.chunk_index,
                pre_rank=pre_rank,
                pre_score=entry.rrf_score,
                post_rank=post_rank,
                post_score=float(score),
            )
            for post_rank, ((pre_rank, entry), score) in enumerate(post_ranked, start=1)
        ]

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        after = [
            {"document": e.metadata.document, "chunk_index": e.chunk_index, "score": e.post_score}
            for e in entries
        ]
        logger.info(
            "reranking",
            extra={"before": before, "after": after, "latency_ms": latency_ms},
        )
        return RerankResult(entries=entries)
