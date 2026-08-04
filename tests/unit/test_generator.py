"""Generation (FR-L3-10)."""

from __future__ import annotations

from bankassist.llm.base import LLMMessage
from bankassist.llm.stub import StubLLMClient
from bankassist.rag.models.retrieval_context import PromptContext
from bankassist.rag.prompts import REFUSAL
from bankassist.rag.stages.generator import Generator


def _context(chunk_count: int) -> PromptContext:
    return PromptContext(
        messages=[LLMMessage(role="system", content="sys"), LLMMessage(role="user", content="usr")],
        chunk_count=chunk_count,
        estimated_tokens=10,
    )


def test_zero_chunks_short_circuits_to_refusal_with_no_llm_call() -> None:
    llm = StubLLMClient(["unused"])

    result = Generator(llm).execute(_context(chunk_count=0))

    assert result.answer == REFUSAL
    assert result.grounded is False
    assert llm.calls == []


def test_grounded_generation_delegates_to_the_llm_client() -> None:
    llm = StubLLMClient(["disputes must be raised within 90 days"])

    result = Generator(llm).execute(_context(chunk_count=1))

    assert result.answer == "disputes must be raised within 90 days"
    assert result.grounded is True
    assert result.latency_ms >= 0.0


def test_model_emitted_refusal_is_detected_structurally() -> None:
    llm = StubLLMClient([REFUSAL])

    result = Generator(llm).execute(_context(chunk_count=1))

    assert result.answer == REFUSAL
    assert result.grounded is False
