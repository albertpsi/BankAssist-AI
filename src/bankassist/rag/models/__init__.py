"""Boundary objects for the RAG pipeline.

The four document metadata fields (``document``, ``title``, ``category``,
``source``) are fixed by the Lab 2 brief and travel unchanged from the loader
through chunking and embedding into the vector store's metadata, so a retrieved
chunk can always name where it came from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """What a chunk needs to cite its origin."""

    document: str
    title: str
    category: str
    source: str


class PolicyDocument(BaseModel):
    """One markdown policy file, paired with its metadata sidecar."""

    metadata: DocumentMetadata
    text: str

    @property
    def document(self) -> str:
        return self.metadata.document


class Chunk(BaseModel):
    """A slice of one document, and where in that document it came from.

    ``char_start``/``char_end`` index into the *normalized* document text, so
    ``normalized[char_start:char_end] == text`` holds.
    """

    metadata: DocumentMetadata
    text: str
    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @property
    def vector_id(self) -> str:
        """A stable id, so re-ingesting a document upserts rather than duplicates."""
        return f"{_slug(self.metadata.document)}#{self.chunk_index}"


class VectorRecord(BaseModel):
    """A chunk plus its embedding, ready for the vector store."""

    id: str
    values: list[float]
    metadata: dict[str, str | int]


class RetrievedChunk(BaseModel):
    """A search hit: the chunk that matched, and how well it matched."""

    text: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    score: float


class RagAnswer(BaseModel):
    """The pipeline's output.

    ``grounded`` is False when the corpus did not support an answer. It is derived
    structurally — empty retrieval, or an answer equal to the refusal constant —
    never by interpreting model prose.
    """

    answer: str
    sources: list[str]
    grounded: bool
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


def _slug(value: str) -> str:
    """Reduce a file name to a stable, id-safe token.

    Not for display — only for building vector ids that survive a file being
    re-ingested from a differently-cased or differently-spaced path.
    """
    kept = [character.lower() if character.isalnum() else "-" for character in value]
    return "".join(kept).strip("-")
