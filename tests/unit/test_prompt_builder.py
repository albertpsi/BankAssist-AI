"""Prompt construction (FR-L3-9)."""

from __future__ import annotations

from bankassist.rag.models import DocumentMetadata
from bankassist.rag.models.rerank_result import RerankEntry, RerankResult
from bankassist.rag.models.retrieval_context import PromptBuildRequest
from bankassist.rag.prompts import REFUSAL
from bankassist.rag.stages.prompt_builder import PromptBuilder


def _entry(document: str, text: str, score: float = 0.9) -> RerankEntry:
    return RerankEntry(
        text=text,
        metadata=DocumentMetadata(
            document=document, title=document, category="Credit Card", source="Test"
        ),
        chunk_index=0,
        pre_rank=1,
        pre_score=0.5,
        post_rank=1,
        post_score=score,
    )


def test_prompt_includes_original_and_rewritten_question_and_context() -> None:
    request = PromptBuildRequest(
        original_question="How long do I have?",
        rewritten_question="What is the chargeback dispute time limit?",
        reranked=RerankResult(entries=[_entry("chargeback.md", "the window is 90 days")]),
    )

    context = PromptBuilder().execute(request)

    user_content = context.messages[-1].content
    assert "How long do I have?" in user_content
    assert "What is the chargeback dispute time limit?" in user_content
    assert "the window is 90 days" in user_content
    assert "chargeback.md" in user_content


def test_grounding_and_refusal_instruction_is_present() -> None:
    request = PromptBuildRequest(
        original_question="q",
        rewritten_question="q",
        reranked=RerankResult(entries=[_entry("a.md", "t")]),
    )

    context = PromptBuilder().execute(request)

    assert REFUSAL in context.messages[0].content


def test_chunk_count_and_token_estimate_are_reported() -> None:
    request = PromptBuildRequest(
        original_question="q",
        rewritten_question="q",
        reranked=RerankResult(entries=[_entry("a.md", "t1"), _entry("b.md", "t2")]),
    )

    context = PromptBuilder().execute(request)

    assert context.chunk_count == 2
    assert context.estimated_tokens > 0


def test_zero_chunks_produces_zero_chunk_count() -> None:
    request = PromptBuildRequest(
        original_question="q", rewritten_question="q", reranked=RerankResult(entries=[])
    )

    context = PromptBuilder().execute(request)

    assert context.chunk_count == 0
