"""Test doubles for the RAG dependencies.

These live in ``src`` rather than ``tests`` for the same reason ``llm/stub.py``
does: they are what lets the entire suite run with no API key, no network, and no
cost (NFR-L2-2), and later labs reuse them rather than reinventing a fake each time.

The embeddings here are deterministic hashes, not meaningful vectors. They exist
to prove wiring, ordering, and dimensionality — never retrieval *quality*, which
only a real model can be judged on.
"""

from __future__ import annotations

import hashlib
import math

from bankassist.rag.models import DocumentMetadata, RetrievedChunk, VectorRecord


class StubEmbedder:
    """Deterministic embeddings derived from the text itself.

    Same text always yields the same vector, and different text almost always
    yields a different one — enough for round-trip and ordering assertions.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
        self.embedded_documents: list[list[str]] = []
        self.embedded_queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_documents.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """Expand a digest of the text into a unit vector of the right width.

        Digested in counter-prefixed blocks rather than tiling one 32-byte hash,
        so two different texts differ across the whole vector instead of
        repeating the same short pattern 48 times.
        """
        data = bytearray()
        block = 0
        while len(data) < self.dimensions:
            data += hashlib.sha256(f"{block}:{text}".encode()).digest()
            block += 1

        raw = [data[index] / 255.0 - 0.5 for index in range(self.dimensions)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


class InMemoryVectorStore:
    """A ``VectorStore`` backed by a dict and honest cosine similarity.

    Upserting by id is what makes the ingestion idempotency test meaningful: a
    re-run overwrites in place here exactly as it does in Pinecone, so the test
    exercises the real property rather than a mock's return value.
    """

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.ensure_index_calls = 0
        self.queries: list[tuple[list[float], int]] = []
        self.index_created = False

    def ensure_index(self) -> bool:
        self.ensure_index_calls += 1
        if self.index_created:
            return False
        self.index_created = True
        return True

    def upsert(self, records: list[VectorRecord]) -> int:
        for record in records:
            self.records[record.id] = record
        return len(records)

    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        self.queries.append((vector, top_k))

        scored = [
            (_cosine(vector, record.values), record)
            for record in self.records.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            RetrievedChunk(
                text=str(record.metadata.get("text", "")),
                metadata=DocumentMetadata(
                    document=str(record.metadata.get("document", "")),
                    title=str(record.metadata.get("title", "")),
                    category=str(record.metadata.get("category", "")),
                    source=str(record.metadata.get("source", "")),
                ),
                chunk_index=int(record.metadata.get("chunk_index", 0)),
                score=score,
            )
            for score, record in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self.records)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
