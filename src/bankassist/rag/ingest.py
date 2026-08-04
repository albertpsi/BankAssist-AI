"""Corpus → chunks → embeddings → vector store.

A library function rather than a script body, so the whole pipeline can be tested
against doubles without spawning a subprocess. ``scripts/ingest_policies.py`` is a
thin CLI over this.
"""

from __future__ import annotations

import time

from bankassist.config import Settings
from bankassist.logging_config import get_logger
from bankassist.rag.chunker import chunk_document
from bankassist.rag.embeddings import Embedder
from bankassist.rag.loader import load_corpus
from bankassist.rag.models import Chunk, PolicyDocument, VectorRecord
from bankassist.rag.vector_store import VectorStore

logger = get_logger(__name__)

# Pinecone metadata values must be scalars, so the chunk text rides along as a
# plain string. It is stored because retrieval must return the text itself —
# there is no second lookup against the source files at query time.
MAX_METADATA_TEXT_CHARS = 4000


class IngestionResult:
    """What one ingestion run did, for the CLI summary and for assertions."""

    def __init__(
        self,
        documents: list[PolicyDocument],
        chunks: list[Chunk],
        vectors_upserted: int,
        index_created: bool,
        elapsed_seconds: float,
    ) -> None:
        self.documents = documents
        self.chunks = chunks
        self.vectors_upserted = vectors_upserted
        self.index_created = index_created
        self.elapsed_seconds = elapsed_seconds


def chunk_corpus(settings: Settings) -> tuple[list[PolicyDocument], list[Chunk]]:
    """Load every document and chunk it. No API call, no cost."""
    documents = load_corpus(settings.markdown_dir, settings.metadata_dir)

    chunks: list[Chunk] = []
    for document in documents:
        document_chunks = chunk_document(
            document,
            size=settings.chunk_size_chars,
            min_size=settings.chunk_min_chars,
            max_size=settings.chunk_max_chars,
            overlap=settings.chunk_overlap_chars,
        )
        logger.info(
            "document chunked",
            extra={
                "document": document.document,
                "category": document.metadata.category,
                "chars": len(document.text),
                "chunk_count": len(document_chunks),
            },
        )
        chunks.extend(document_chunks)

    logger.info("chunking complete", extra={"chunk_count": len(chunks)})
    return documents, chunks


def to_vector_records(chunks: list[Chunk], vectors: list[list[float]]) -> list[VectorRecord]:
    """Pair chunks with their embeddings.

    Raises:
        ValueError: the counts disagree, which would silently mis-attribute every
            chunk after the first mismatch.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    return [
        VectorRecord(
            id=chunk.vector_id,
            values=vector,
            metadata={
                "document": chunk.metadata.document,
                "title": chunk.metadata.title,
                "category": chunk.metadata.category,
                "source": chunk.metadata.source,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text[:MAX_METADATA_TEXT_CHARS],
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def run_ingestion(
    settings: Settings,
    embedder: Embedder,
    store: VectorStore,
) -> IngestionResult:
    """Run the full pipeline.

    Idempotent: vector ids are derived from the document name and chunk index, so
    a second run overwrites the first rather than adding a duplicate corpus.
    """
    started = time.perf_counter()
    logger.info(
        "ingestion started",
        extra={
            "corpus_dir": str(settings.markdown_dir),
            "embedding_model": settings.embedding_model,
            "index": settings.pinecone_index_name,
            "namespace": settings.pinecone_namespace,
        },
    )

    documents, chunks = chunk_corpus(settings)

    index_created = store.ensure_index()

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    records = to_vector_records(chunks, vectors)
    upserted = store.upsert(records)

    elapsed = time.perf_counter() - started
    logger.info(
        "ingestion complete",
        extra={
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "vectors_upserted": upserted,
            "index_created": index_created,
            "elapsed_ms": round(elapsed * 1000.0, 2),
        },
    )

    return IngestionResult(documents, chunks, upserted, index_created, elapsed)
