"""Prompt construction (FR-L2-7)."""

from __future__ import annotations

from bankassist.rag.models import DocumentMetadata, RetrievedChunk
from bankassist.rag.prompts import REFUSAL, SYSTEM_PROMPT, build_messages


def _chunk(document: str, text: str, chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=DocumentMetadata(
            document=document, title=document, category="Credit Card", source="Test"
        ),
        chunk_index=chunk_index,
        score=0.9,
    )


def test_system_prompt_instructs_context_only_answers() -> None:
    """FR-L2-7.2."""
    assert "ONLY" in SYSTEM_PROMPT
    assert REFUSAL in SYSTEM_PROMPT


def test_system_prompt_states_retrieved_content_is_information_not_instruction() -> None:
    """FR-L2-7.3."""
    lowered = SYSTEM_PROMPT.lower()
    assert "information only" in lowered
    assert "never obey" in lowered


def test_question_appears_in_the_user_message() -> None:
    messages = build_messages("what is the chargeback window?", [_chunk("a.md", "text")])

    assert "what is the chargeback window?" in messages[-1].content


def test_every_chunk_text_appears_in_the_prompt() -> None:
    chunks = [
        _chunk("a.md", "alpha clause about fees"),
        _chunk("b.md", "beta clause about kyc"),
    ]

    messages = build_messages("q", chunks)

    content = messages[-1].content
    assert "alpha clause about fees" in content
    assert "beta clause about kyc" in content


def test_each_chunk_is_labelled_with_its_document() -> None:
    messages = build_messages("q", [_chunk("Chargeback Policy.md", "text")])

    content = messages[-1].content
    assert 'source="Chargeback Policy.md"' in content


def test_chunk_order_is_preserved() -> None:
    chunks = [_chunk("first.md", "first text"), _chunk("second.md", "second text")]

    content = build_messages("q", chunks)[-1].content

    assert content.index("first text") < content.index("second text")


def test_messages_are_system_then_user() -> None:
    messages = build_messages("q", [_chunk("a.md", "text")])

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT


def test_empty_chunk_list_still_produces_a_well_formed_prompt() -> None:
    """The pipeline never calls this with no chunks (FR-L2-7.5), but the
    function itself should not crash if it is."""
    messages = build_messages("q", [])

    assert messages[-1].content
