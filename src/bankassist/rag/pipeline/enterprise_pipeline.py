"""The enterprise RAG pipeline: orchestration only (FR-L3-1.3).

Every stage is constructed elsewhere and injected here. This module contains no
retrieval, ranking, filtering, or prompt logic of its own — it sequences the
stages under ``rag/stages/`` and assembles the ``PipelineResult``.
"""

from __future__ import annotations

import time

from bankassist.config import Settings
from bankassist.observability import run as observability_run
from bankassist.rag.interfaces.classifier import Classifier
from bankassist.rag.interfaces.reranker import Reranker
from bankassist.rag.interfaces.retriever import Retriever
from bankassist.rag.models.pipeline_result import PipelineResult
from bankassist.rag.models.rerank_result import RerankEntry
from bankassist.rag.models.retrieval_context import MetadataFilters, PromptBuildRequest
from bankassist.rag.stages.generator import Generator
from bankassist.rag.stages.hybrid_retriever import HybridRetriever
from bankassist.rag.stages.metadata_filter import MetadataFilter
from bankassist.rag.stages.prompt_builder import PromptBuilder
from bankassist.rag.stages.query_rewriter import QueryRewriter
from bankassist.rag.stages.rrf_ranker import RRFRanker


def _distinct_sources(entries: list[RerankEntry]) -> list[str]:
    """Document names, deduplicated, in reranked order (FR-L3-11.2)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        if entry.metadata.document not in seen:
            seen.add(entry.metadata.document)
            ordered.append(entry.metadata.document)
    return ordered


class EnterpriseRagPipeline:
    """Classify → rewrite → hybrid retrieve → filter → RRF → rerank → prompt → generate."""

    def __init__(
        self,
        settings: Settings,
        classifier: Classifier,
        rewriter: QueryRewriter,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        hybrid_retriever: HybridRetriever | None = None,
        metadata_filter: MetadataFilter | None = None,
        rrf_ranker: RRFRanker | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._rewriter = rewriter
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._hybrid_retriever = hybrid_retriever or HybridRetriever()
        self._metadata_filter = metadata_filter or MetadataFilter()
        self._rrf_ranker = rrf_ranker or RRFRanker(k=settings.rrf_k)
        self._reranker = reranker
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._generator = generator

    def answer(self, question: str, filters: MetadataFilters | None = None) -> PipelineResult:
        latencies: dict[str, float] = {}

        # Each stage gets a named AgentOps operation span (Lab 6 requirements
        # §5/§6): this whole method runs inside one LangGraph node
        # (policy_agent/dispute_agent), so automatic LangGraph instrumentation
        # sees exactly one step here without these — the multi-stage pipeline
        # would otherwise be invisible in the AgentOps trace.
        classification = observability_run(
            "operation", "rag.classify", self._classifier.execute, question
        )
        latencies["classification"] = classification.latency_ms

        rewrite = observability_run(
            "operation", "rag.rewrite", self._rewriter.execute, question, classification
        )
        latencies["rewrite"] = rewrite.latency_ms

        vector_top_k = self._settings.retrieval_vector_top_k_enterprise
        bm25_top_k = self._settings.retrieval_bm25_top_k
        vector_results = observability_run(
            "operation",
            "rag.vector_retrieval",
            self._vector_retriever.execute,
            rewrite.rewritten_question,
            top_k=vector_top_k,
        )
        latencies["vector_retrieval"] = vector_results.latency_ms

        bm25_results = observability_run(
            "operation",
            "rag.bm25_retrieval",
            self._bm25_retriever.execute,
            rewrite.rewritten_question,
            top_k=bm25_top_k,
        )
        latencies["bm25_retrieval"] = bm25_results.latency_ms

        hybrid = self._hybrid_retriever.execute(vector_results, bm25_results)
        filtered = self._metadata_filter.execute(hybrid, filters)

        started_rrf = time.perf_counter()
        rrf_result = observability_run("operation", "rag.rrf", self._rrf_ranker.execute, filtered)
        latencies["rrf"] = round((time.perf_counter() - started_rrf) * 1000.0, 2)

        candidates = rrf_result.model_copy(
            update={"entries": rrf_result.entries[: self._settings.rerank_candidate_count]}
        )
        started_rerank = time.perf_counter()
        reranked = observability_run(
            "operation",
            "rag.rerank",
            self._reranker.execute,
            rewrite.rewritten_question,
            candidates,
            top_n=self._settings.rerank_top_n,
        )
        latencies["rerank"] = round((time.perf_counter() - started_rerank) * 1000.0, 2)

        prompt_context = self._prompt_builder.execute(
            PromptBuildRequest(
                original_question=question,
                rewritten_question=rewrite.rewritten_question,
                reranked=reranked,
            )
        )

        generation = observability_run(
            "operation", "rag.generate", self._generator.execute, prompt_context
        )
        latencies["generation"] = generation.latency_ms

        citations = _distinct_sources(reranked.entries) if generation.grounded else []

        return PipelineResult(
            original_question=question,
            classification=classification,
            rewritten_question=rewrite.rewritten_question,
            vector_results=vector_results,
            bm25_results=bm25_results,
            rrf_results=rrf_result,
            reranked_results=reranked,
            prompt_context=prompt_context,
            generated_answer=generation.answer,
            grounded=generation.grounded,
            citations=citations,
            latencies=latencies,
        )
