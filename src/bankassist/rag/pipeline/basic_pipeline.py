"""The basic RAG pipeline: retrieve, then answer.

Plain vector similarity only — no hybrid search, no metadata filter, no RRF, no
reranking, no query rewriting. Those are Lab 3 and are added as a second pipeline
alongside this one, not folded into it, so `basic` and `enterprise` stay
separately selectable (FR-3.8).
"""

from __future__ import annotations

import time

from bankassist.config import Settings
from bankassist.llm.base import LLMClient
from bankassist.logging_config import get_logger
from bankassist.rag.embeddings import Embedder
from bankassist.rag.models import RagAnswer, RetrievedChunk
from bankassist.rag.prompts import REFUSAL, build_messages
from bankassist.rag.vector_store import VectorStore

logger = get_logger(__name__)


class BasicRagPipeline:
    """Retrieve, then answer, grounded in what was retrieved."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed the question and return the nearest chunks, best first.

        FR-L2-6.2: top_k defaults to the configured value (5) when not overridden.
        A blank question still makes the round trip — validating input shape is
        the API layer's job (FR-L2-9.2), not the pipeline's.
        """
        k = self._settings.retrieval_top_k if top_k is None else top_k

        vector = self._embedder.embed_query(question)
        results = self._store.query(vector, top_k=k)

        logger.info(
            "pipeline retrieval",
            extra={"question_length": len(question), "top_k": k, "result_count": len(results)},
        )
        return results

    def answer(self, question: str) -> RagAnswer:
        """Retrieve, then generate a grounded answer with source citations.

        FR-L2-7.5: zero retrieved chunks refuses deterministically, with no LLM
        call — there is nothing for the model to be grounded in, so asking it
        would only invite it to fall back on parametric knowledge.
        """
        started = time.perf_counter()
        chunks = self.retrieve(question)

        if not chunks:
            return self._log_answer(question, self._refusal(chunks=[]), started)

        messages = build_messages(question, chunks)
        response = self._llm.complete(messages)
        text = response.text.strip()

        # FR-L2-7.4/8.3: a model-emitted refusal is detected structurally — an
        # exact match against the constant, never by interpreting prose — and
        # empties the source list, since nothing retrieved was actually used.
        if text == REFUSAL:
            return self._log_answer(question, self._refusal(chunks=chunks), started)

        sources = _distinct_sources(chunks)
        result = RagAnswer(answer=text, sources=sources, grounded=True, retrieved=chunks)
        return self._log_answer(question, result, started)

    def _refusal(self, *, chunks: list[RetrievedChunk]) -> RagAnswer:
        return RagAnswer(answer=REFUSAL, sources=[], grounded=False, retrieved=chunks)

    def _log_answer(self, question: str, result: RagAnswer, started: float) -> RagAnswer:
        logger.info(
            "pipeline answer",
            extra={
                "question_length": len(question),
                "grounded": result.grounded,
                "source_count": len(result.sources),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
        )
        return result


def _distinct_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Document names, deduplicated, in first-retrieved order (FR-L2-8.1)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        if chunk.metadata.document not in seen:
            seen.add(chunk.metadata.document)
            ordered.append(chunk.metadata.document)
    return ordered
