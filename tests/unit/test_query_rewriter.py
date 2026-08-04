"""Query rewriting (FR-L3-4)."""

from __future__ import annotations

from bankassist.llm.stub import StubLLMClient
from bankassist.rag.models.classification_result import ClassificationResult
from bankassist.rag.stages.query_rewriter import QueryRewriter


def _classification() -> ClassificationResult:
    return ClassificationResult(label="Procedure", confidence=0.9)


def test_original_question_is_always_preserved() -> None:
    llm = StubLLMClient(["What is the chargeback dispute time limit?"])

    result = QueryRewriter(llm).execute("How long do I have?", _classification())

    assert result.original_question == "How long do I have?"
    assert result.rewritten_question == "What is the chargeback dispute time limit?"
    assert result.fallback_used is False


def test_empty_rewrite_falls_back_to_the_original() -> None:
    llm = StubLLMClient([""])

    result = QueryRewriter(llm).execute("How long do I have?", _classification())

    assert result.rewritten_question == "How long do I have?"
    assert result.fallback_used is True


def test_llm_failure_falls_back_to_the_original_rather_than_raising() -> None:
    llm = StubLLMClient([])  # exhausted immediately -> LLMError inside the stage

    result = QueryRewriter(llm).execute("How long do I have?", _classification())

    assert result.rewritten_question == "How long do I have?"
    assert result.fallback_used is True


def test_latency_is_recorded() -> None:
    llm = StubLLMClient(["rewritten"])

    result = QueryRewriter(llm).execute("q", _classification())

    assert result.latency_ms >= 0.0
