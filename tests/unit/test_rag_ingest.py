"""Ingestion pipeline (FR-L2-5, NFR-L2-3).

Runs the real pipeline against an in-memory store and a stub embedder, so
idempotency is exercised as a genuine upsert-by-id property rather than asserted
against a mock's canned return value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bankassist.config import Settings
from bankassist.rag.ingest import chunk_corpus, run_ingestion, to_vector_records
from bankassist.rag.models import Chunk, DocumentMetadata
from bankassist.rag.stubs import InMemoryVectorStore, StubEmbedder

SIDECAR = {
    "title": "Transaction Dispute Form",
    "category": "Credit Card",
    "source": "Official SBI Card",
    "url": "https://example.invalid/form.pdf",
}


@pytest.fixture
def corpus_settings(tmp_path: Path, settings: Settings) -> Settings:
    """Settings pointed at a small generated corpus."""
    markdown_dir = tmp_path / "markdown"
    metadata_dir = tmp_path / "metadata"
    markdown_dir.mkdir()
    metadata_dir.mkdir()

    for stem, paragraphs in (("01_Alpha", 8), ("02_Beta", 3)):
        body = "\n\n".join(
            f"Clause {index}. " + "policy detail " * 40 for index in range(paragraphs)
        )
        (markdown_dir / f"{stem}.md").write_text(body, encoding="utf-8")
        (metadata_dir / f"{stem}.json").write_text(json.dumps(SIDECAR), encoding="utf-8")

    return settings.model_copy(update={"policy_corpus_dir": tmp_path, "embedding_batch_size": 4})


def _chunk(document: str, index: int, text: str = "body") -> Chunk:
    return Chunk(
        metadata=DocumentMetadata(
            document=document, title="T", category="C", source="S"
        ),
        text=text,
        chunk_index=index,
        char_start=0,
        char_end=len(text),
    )


def test_chunk_corpus_returns_documents_and_chunks(corpus_settings: Settings) -> None:
    documents, chunks = chunk_corpus(corpus_settings)

    assert len(documents) == 2
    assert len(chunks) > len(documents)
    assert {chunk.metadata.document for chunk in chunks} == {"01_Alpha.md", "02_Beta.md"}


def test_ingestion_upserts_one_vector_per_chunk(corpus_settings: Settings) -> None:
    store = InMemoryVectorStore()

    result = run_ingestion(corpus_settings, StubEmbedder(dimensions=8), store)

    assert result.vectors_upserted == len(result.chunks)
    assert store.count() == len(result.chunks)


def test_index_is_ensured_before_anything_is_written(corpus_settings: Settings) -> None:
    """FR-L2-5.2: upserting into an index that does not exist yet is a hard failure."""
    store = InMemoryVectorStore()

    result = run_ingestion(corpus_settings, StubEmbedder(dimensions=8), store)

    assert store.ensure_index_calls == 1
    assert result.index_created is True


def test_rerunning_ingestion_does_not_duplicate_the_corpus(corpus_settings: Settings) -> None:
    """NFR-L2-3: deterministic ids make a second run an overwrite, not a copy."""
    store = InMemoryVectorStore()

    first = run_ingestion(corpus_settings, StubEmbedder(dimensions=8), store)
    count_after_first = store.count()
    second = run_ingestion(corpus_settings, StubEmbedder(dimensions=8), store)

    assert count_after_first == store.count()
    assert first.vectors_upserted == second.vectors_upserted
    assert second.index_created is False


def test_vector_ids_are_deterministic_across_runs(corpus_settings: Settings) -> None:
    first = InMemoryVectorStore()
    second = InMemoryVectorStore()

    run_ingestion(corpus_settings, StubEmbedder(dimensions=8), first)
    run_ingestion(corpus_settings, StubEmbedder(dimensions=8), second)

    assert set(first.records) == set(second.records)


def test_stored_metadata_carries_the_four_fields_and_the_text() -> None:
    """FR-L2-5.3: retrieval returns chunk text from the store, not a second file read."""
    chunks = [_chunk("01_Alpha.md", 0, "the chargeback window is 90 days")]

    records = to_vector_records(chunks, [[0.1, 0.2]])

    metadata = records[0].metadata
    assert metadata["document"] == "01_Alpha.md"
    assert metadata["title"] == "T"
    assert metadata["category"] == "C"
    assert metadata["source"] == "S"
    assert metadata["chunk_index"] == 0
    assert metadata["text"] == "the chargeback window is 90 days"


def test_vector_id_combines_document_and_chunk_index() -> None:
    chunks = [_chunk("03_Raising_Card_Dispute.md", 7)]

    records = to_vector_records(chunks, [[0.0]])

    assert records[0].id == "03-raising-card-dispute-md#7"


def test_chunks_from_different_documents_never_collide() -> None:
    chunks = [_chunk("01_Alpha.md", 0), _chunk("02_Beta.md", 0)]

    records = to_vector_records(chunks, [[0.0], [1.0]])

    assert records[0].id != records[1].id


def test_mismatched_vector_count_is_rejected() -> None:
    """A silent mis-pairing would corrupt every retrieval, undetectably."""
    chunks = [_chunk("01_Alpha.md", 0), _chunk("01_Alpha.md", 1)]

    with pytest.raises(ValueError, match="2 chunks but 1 vectors"):
        to_vector_records(chunks, [[0.0]])


def test_oversized_chunk_text_is_truncated_for_metadata() -> None:
    """Pinecone caps metadata size; a 40 KB chunk must not fail the whole upsert."""
    chunks = [_chunk("01_Alpha.md", 0, "x" * 9000)]

    records = to_vector_records(chunks, [[0.0]])

    assert len(str(records[0].metadata["text"])) == 4000


def test_embedder_receives_chunk_text_in_order(corpus_settings: Settings) -> None:
    embedder = StubEmbedder(dimensions=8)

    result = run_ingestion(corpus_settings, embedder, InMemoryVectorStore())

    embedded = [text for batch in embedder.embedded_documents for text in batch]
    assert embedded == [chunk.text for chunk in result.chunks]
