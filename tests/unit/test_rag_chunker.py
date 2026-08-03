"""Chunking behaviour (FR-L2-3).

The chunker is pure, so these tests construct text with known break points rather
than reading the corpus — a failure here should point at the algorithm, not at a
document someone edited.
"""

from __future__ import annotations

import pytest

from bankassist.rag.chunker import chunk_document, chunk_spans
from bankassist.rag.models import DocumentMetadata, PolicyDocument

SIZE = 800
MIN_SIZE = 700
MAX_SIZE = 900
OVERLAP = 120


def split(text: str, **overrides: int) -> list[tuple[int, int]]:
    kwargs = {"size": SIZE, "min_size": MIN_SIZE, "max_size": MAX_SIZE, "overlap": OVERLAP}
    kwargs.update(overrides)
    return chunk_spans(text, **kwargs)  # type: ignore[arg-type]


def document(text: str) -> PolicyDocument:
    return PolicyDocument(
        metadata=DocumentMetadata(
            document="Test Policy.md",
            title="Test Policy",
            category="Credit Card",
            source="Test Source",
        ),
        text=text,
    )


def test_short_document_is_one_chunk() -> None:
    text = "The chargeback window is 90 days."

    spans = split(text)

    assert spans == [(0, len(text))]


@pytest.mark.parametrize("blank", ["", "   ", "\n\n\n", "\t \n \t"])
def test_empty_or_whitespace_document_yields_no_chunks(blank: str) -> None:
    assert split(blank) == []


def test_non_final_chunks_respect_the_size_window() -> None:
    """FR-L2-3.2: every chunk but the last lands within [min, max]."""
    text = " ".join(f"sentence number {index}." for index in range(1200))

    spans = split(text)

    assert len(spans) > 3
    for start, end in spans[:-1]:
        assert MIN_SIZE <= end - start <= MAX_SIZE
    assert spans[-1][1] - spans[-1][0] <= MAX_SIZE


def test_overlap_text_is_carried_into_the_next_chunk() -> None:
    """FR-L2-3.3: the overlap is real shared text, not just an index arithmetic."""
    text = " ".join(f"word{index}" for index in range(2000))

    spans = split(text)

    assert len(spans) >= 3
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start < previous_end, "consecutive chunks must overlap"
        shared = previous_end - next_start
        # The cut is pulled back by exactly `overlap`, then trimmed forward off
        # any whitespace, so the shared region is at most the configured overlap.
        assert 0 < shared <= OVERLAP


def test_spans_slice_back_to_their_own_text() -> None:
    """FR-L2-3.6: `text[start:end]` is the chunk — the audit trail must hold."""
    text = "\n\n".join(f"Paragraph {index} about fees and charges." for index in range(200))

    doc = document(text)
    chunks = chunk_document(doc, size=SIZE, min_size=MIN_SIZE, max_size=MAX_SIZE, overlap=OVERLAP)

    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


def test_paragraph_boundary_is_preferred_over_mid_sentence() -> None:
    """FR-L2-3.4: with a blank line inside the window, the cut lands on it."""
    paragraph = "A" * 750
    text = paragraph + "\n\n" + "B" * 900

    spans = split(text)

    assert spans[0] == (0, 750)
    assert text[spans[1][0]] == "B" or spans[1][0] < 750


def test_unbreakable_text_hard_cuts_at_max_size() -> None:
    """FR-L2-3.4/3.5: no whitespace anywhere must still terminate, at the max."""
    text = "x" * 5000

    spans = split(text)

    assert spans[0] == (0, MAX_SIZE)
    assert spans[-1][1] == 5000
    for start, end in spans[:-1]:
        assert end - start == MAX_SIZE


def test_chunking_is_deterministic() -> None:
    text = "\n\n".join(f"Clause {index}. " + "detail " * 40 for index in range(60))

    assert split(text) == split(text)


def test_chunks_never_start_or_end_on_whitespace() -> None:
    text = "\n\n".join("   " + "padded clause text " * 45 + "   " for _ in range(30))

    spans = split(text)

    assert spans
    for start, end in spans:
        assert not text[start].isspace()
        assert not text[end - 1].isspace()


def test_progress_is_guaranteed_when_overlap_exceeds_min_size() -> None:
    """FR-L2-3.5: settings validation forbids this, but a direct caller could
    still do it — and an infinite loop is a hang, not a test failure."""
    text = "word " * 2000

    spans = split(text, overlap=MAX_SIZE + 500)

    assert spans
    starts = [start for start, _ in spans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_chunk_metadata_and_index_are_attached() -> None:
    """FR-L2-2.2: metadata rides along unchanged, indexes are 0-based and dense."""
    doc = document("\n\n".join(f"Section {index} " + "body " * 60 for index in range(40)))

    chunks = chunk_document(doc, size=SIZE, min_size=MIN_SIZE, max_size=MAX_SIZE, overlap=OVERLAP)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.metadata == doc.metadata


def test_vector_id_is_stable_and_id_safe() -> None:
    """FR-L2-5.4: the id is what makes re-ingestion an upsert."""
    doc = document("Only one short chunk here.")

    chunks = chunk_document(doc, size=SIZE, min_size=MIN_SIZE, max_size=MAX_SIZE, overlap=OVERLAP)

    assert chunks[0].vector_id == "test-policy-md#0"
