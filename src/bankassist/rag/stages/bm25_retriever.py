"""Sparse (BM25) retrieval leg of hybrid search (FR-L3-5.2).

Built once, in-process, from the same chunked corpus Lab 2 already ingests into
Pinecone — no second chunking path (design decision, lab-03 spec §5.5).
"""

from __future__ import annotations

import re
import time

from rank_bm25 import BM25Okapi

from bankassist.logging_config import get_logger
from bankassist.rag.models import Chunk
from bankassist.rag.models.retrieval_result import RetrievalResult, ScoredChunk

logger = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokenization — good enough for BM25."""
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """An in-process BM25Okapi index over a fixed set of chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        self._index = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def execute(self, query: str, top_k: int = 20) -> RetrievalResult:
        started = time.perf_counter()

        if self._index is None or not self._chunks:
            results: list[ScoredChunk] = []
        else:
            scores = self._index.get_scores(tokenize(query))
            ranked = sorted(
                zip(scores, self._chunks, strict=True), key=lambda pair: pair[0], reverse=True
            )
            results = [
                ScoredChunk(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                )
                for score, chunk in ranked[:top_k]
                # Exact zero means none of the query's tokens appear in this
                # chunk at all. A small or tiny-corpus-degenerate *negative*
                # score (BM25's idf can go negative when a term appears in
                # every document — see rank_bm25's epsilon handling) still
                # means the terms matched, so it is kept and ranked.
                if score != 0.0
            ]

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        logger.info(
            "bm25 retrieval",
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
