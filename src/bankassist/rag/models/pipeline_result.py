"""The enterprise pipeline's full, explainable output (FR-L3-11).

Designed to become Lab 6 AgentOps input unmodified — every field a trace would
need is already present and typed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from bankassist.rag.models.classification_result import ClassificationResult
from bankassist.rag.models.rerank_result import RerankResult
from bankassist.rag.models.retrieval_context import PromptContext
from bankassist.rag.models.retrieval_result import RetrievalResult, RRFResult


class PipelineResult(BaseModel):
    """Every intermediate output of one enterprise-mode request."""

    original_question: str
    classification: ClassificationResult
    rewritten_question: str
    vector_results: RetrievalResult
    bm25_results: RetrievalResult
    rrf_results: RRFResult
    reranked_results: RerankResult
    prompt_context: PromptContext
    generated_answer: str
    grounded: bool
    citations: list[str] = Field(default_factory=list)
    latencies: dict[str, float] = Field(default_factory=dict)
