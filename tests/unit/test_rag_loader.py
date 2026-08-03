"""Corpus loading and metadata extraction (FR-L2-1, FR-L2-2).

Every test builds its own corpus in a tmp_path, so the suite never depends on the
real documents staying byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bankassist.errors import IngestionError
from bankassist.rag.loader import load_corpus, normalize_text

SIDECAR = {
    "id": "03",
    "title": "Transaction Dispute Form",
    "category": "Credit Card",
    "source": "Official SBI Card",
    "url": "https://example.invalid/form.pdf",
    "document_type": "Form",
    "effective_date": None,
}


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal well-formed corpus: one markdown file, one sidecar."""
    markdown_dir = tmp_path / "markdown"
    metadata_dir = tmp_path / "metadata"
    markdown_dir.mkdir()
    metadata_dir.mkdir()

    (markdown_dir / "03_Raising_Card_Dispute.md").write_text(
        "# Transaction Dispute Form\n\nDisputes must be raised within 90 days.",
        encoding="utf-8",
    )
    (metadata_dir / "03_Raising_Card_Dispute.json").write_text(
        json.dumps(SIDECAR), encoding="utf-8"
    )
    return markdown_dir, metadata_dir


def test_extracts_the_four_required_fields(corpus: tuple[Path, Path]) -> None:
    """FR-L2-2.1: document from the file name, the rest from the sidecar."""
    documents = load_corpus(*corpus)

    assert len(documents) == 1
    metadata = documents[0].metadata
    assert metadata.document == "03_Raising_Card_Dispute.md"
    assert metadata.title == "Transaction Dispute Form"
    assert metadata.category == "Credit Card"
    assert metadata.source == "Official SBI Card"


def test_extra_sidecar_keys_are_not_stored(corpus: tuple[Path, Path]) -> None:
    """FR-L2-2.3: url/document_type/effective_date are Lab 3 filters, not Lab 2 metadata."""
    metadata = load_corpus(*corpus)[0].metadata

    stored = metadata.model_dump()
    assert set(stored) == {"document", "title", "category", "source"}


def test_documents_are_loaded_in_sorted_order(corpus: tuple[Path, Path]) -> None:
    markdown_dir, metadata_dir = corpus
    for stem in ("01_Alpha", "02_Beta"):
        (markdown_dir / f"{stem}.md").write_text(f"Body of {stem}.", encoding="utf-8")
        (metadata_dir / f"{stem}.json").write_text(json.dumps(SIDECAR), encoding="utf-8")

    documents = load_corpus(markdown_dir, metadata_dir)

    names = [doc.metadata.document for doc in documents]
    assert names == sorted(names)


def test_non_markdown_files_are_ignored(corpus: tuple[Path, Path]) -> None:
    markdown_dir, metadata_dir = corpus
    (markdown_dir / "notes.txt").write_text("not part of the corpus", encoding="utf-8")
    (markdown_dir / "03_Raising_Card_Dispute.pdf").write_bytes(b"%PDF-1.4")

    assert len(load_corpus(markdown_dir, metadata_dir)) == 1


def test_missing_sidecar_names_the_document(corpus: tuple[Path, Path]) -> None:
    """FR-L2-1.2: an uncited chunk is worse than a failed ingest."""
    markdown_dir, metadata_dir = corpus
    (markdown_dir / "99_Orphan.md").write_text("No sidecar for this one.", encoding="utf-8")

    with pytest.raises(IngestionError) as excinfo:
        load_corpus(markdown_dir, metadata_dir)

    assert "99_Orphan.md" in excinfo.value.message
    assert excinfo.value.details["document"] == "99_Orphan.md"


def test_unparseable_sidecar_names_the_document(corpus: tuple[Path, Path]) -> None:
    markdown_dir, metadata_dir = corpus
    (markdown_dir / "98_Broken.md").write_text("Body.", encoding="utf-8")
    (metadata_dir / "98_Broken.json").write_text("{not json,", encoding="utf-8")

    with pytest.raises(IngestionError, match="98_Broken.json"):
        load_corpus(markdown_dir, metadata_dir)


@pytest.mark.parametrize("key", ["title", "category", "source"])
def test_missing_required_key_is_rejected(corpus: tuple[Path, Path], key: str) -> None:
    markdown_dir, metadata_dir = corpus
    incomplete = {other: value for other, value in SIDECAR.items() if other != key}
    (markdown_dir / "97_Incomplete.md").write_text("Body.", encoding="utf-8")
    (metadata_dir / "97_Incomplete.json").write_text(json.dumps(incomplete), encoding="utf-8")

    with pytest.raises(IngestionError) as excinfo:
        load_corpus(markdown_dir, metadata_dir)

    assert key in excinfo.value.details["missing"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_required_value_is_treated_as_missing(
    corpus: tuple[Path, Path], blank: str
) -> None:
    markdown_dir, metadata_dir = corpus
    (markdown_dir / "96_Blank.md").write_text("Body.", encoding="utf-8")
    (metadata_dir / "96_Blank.json").write_text(
        json.dumps({**SIDECAR, "category": blank}), encoding="utf-8"
    )

    with pytest.raises(IngestionError, match="category"):
        load_corpus(markdown_dir, metadata_dir)


def test_empty_document_is_rejected(corpus: tuple[Path, Path]) -> None:
    markdown_dir, metadata_dir = corpus
    (markdown_dir / "95_Empty.md").write_text("   \n\n  ", encoding="utf-8")
    (metadata_dir / "95_Empty.json").write_text(json.dumps(SIDECAR), encoding="utf-8")

    with pytest.raises(IngestionError, match="empty"):
        load_corpus(markdown_dir, metadata_dir)


def test_missing_directory_is_reported_not_silently_empty(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="markdown directory not found"):
        load_corpus(tmp_path / "nope", tmp_path)


def test_directory_with_no_documents_is_reported(tmp_path: Path) -> None:
    (tmp_path / "markdown").mkdir()
    (tmp_path / "metadata").mkdir()

    with pytest.raises(IngestionError, match="No markdown documents"):
        load_corpus(tmp_path / "markdown", tmp_path / "metadata")


class TestNormalizeText:
    """FR-L2-3.1a: whitespace only. PDF-extraction artifacts stay put."""

    def test_line_endings_are_unified(self) -> None:
        assert normalize_text("a\r\nb\rc") == "a\nb\nc"

    def test_trailing_spaces_are_stripped(self) -> None:
        assert normalize_text("fees   \n  charges  ") == "fees\n  charges"

    def test_long_blank_runs_collapse_to_one_gap(self) -> None:
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_paragraph_breaks_are_preserved(self) -> None:
        assert normalize_text("a\n\nb") == "a\n\nb"

    def test_pdf_layout_artifacts_are_left_alone(self) -> None:
        """Repairing these is layout-aware parsing, which Lab 2 excludes."""
        raw = "1 | P a g e K Y C P o l i c y\n\n## Transaction\n\n## Date"

        assert normalize_text(raw) == raw
