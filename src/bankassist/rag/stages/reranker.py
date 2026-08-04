"""Cross-encoder reranking (FR-L3-8, ADR-0008).

``cross-encoder/ms-marco-MiniLM-L-6-v2`` loads once per process — construction
is expensive (model download + weight load), so callers should build one
``CrossEncoderReranker`` and reuse it, the same lazy-singleton pattern
``api/routes/rag.py::get_pipeline`` already uses for the Pinecone client.

**Selection is rank-fused, not cross-encoder-score-alone** (amended 2026-08-04).
Investigating a real grounding failure (the KYC "list of OVDs" chunk) showed the
small MiniLM cross-encoder can rank a terse, correct, list-formatted chunk far
below denser boilerplate/preamble chunks from the same document — while RRF
fusion (vector + BM25) still placed it respectably. Trusting the cross-encoder
score alone let a single noisy model judgement veto a chunk two independent
retrieval signals had already surfaced. Selection now reciprocal-rank-fuses the
RRF pre-rank with the cross-encoder rank (the same RRF technique
``rrf_ranker.py`` already uses for vector+BM25, applied one level up), so a
chunk has to be doubly unranked — by fusion *and* by the reranker — to be
dropped. ``post_score`` still reports the raw cross-encoder score for
explainability; only the selection/ordering criterion changed.
"""

from __future__ import annotations

import time

from bankassist.logging_config import get_logger
from bankassist.rag.models.rerank_result import RerankEntry, RerankResult
from bankassist.rag.models.retrieval_result import RRFResult

logger = get_logger(__name__)


def fuse_ranks(scores: list[float], *, fusion_k: int = 60) -> list[tuple[int, int, float]]:
    """Reciprocal-rank-fuse input order (RRF pre-rank) with score order (CE rank).

    Pure function, no model — independently unit-testable without loading the
    real cross-encoder (NFR-L3-2). ``scores[i]`` is the cross-encoder score for
    the candidate at input position ``i`` (0-indexed; input position 0 = RRF
    rank 1, the fused vector+BM25 ranking's strongest match).

    Returns ``(fusion_score, original_index, ce_score)`` tuples sorted
    descending by ``fusion_score`` — highest-ranked-by-both-signals first.
    """
    ce_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ce_rank_by_index = {index: rank for rank, index in enumerate(ce_order, start=1)}

    fused = []
    for index, score in enumerate(scores):
        rrf_rank = index + 1
        ce_rank = ce_rank_by_index[index]
        fusion_score = 1.0 / (fusion_k + rrf_rank) + 1.0 / (fusion_k + ce_rank)
        fused.append((fusion_score, index, score))
    fused.sort(key=lambda item: item[0], reverse=True)
    return fused


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, fusion_k: int = 60) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        # Same constant family as `RRFRanker.k` (config's `rrf_k`, default 60) —
        # not wired to `Settings` here to keep this stage's constructor to just
        # the model id, matching how every other stage takes its one dependency.
        self._fusion_k = fusion_k

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

        top = fuse_ranks([float(s) for s in scores], fusion_k=self._fusion_k)[:top_n]

        entries = [
            RerankEntry(
                text=candidates[index].text,
                metadata=candidates[index].metadata,
                chunk_index=candidates[index].chunk_index,
                pre_rank=index + 1,
                pre_score=candidates[index].rrf_score,
                post_rank=post_rank,
                post_score=ce_score,
            )
            for post_rank, (_, index, ce_score) in enumerate(top, start=1)
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
