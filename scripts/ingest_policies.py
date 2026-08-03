"""Ingest the banking policy corpus into Pinecone.

    python scripts/ingest_policies.py --dry-run       chunk only, no API calls
    python scripts/ingest_policies.py --embed-only    chunk and embed, no upsert
    python scripts/ingest_policies.py                 chunk, embed, and upsert
    python scripts/ingest_policies.py --search "..."  retrieval only, existing index
    python scripts/ingest_policies.py --ask "..."     full pipeline: retrieve + answer

Deliberately thin: everything it does lives in ``bankassist.rag`` so it can be
tested without a subprocess. This file owns argument parsing and the human-facing
summary, nothing else.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Allow `python scripts/ingest_policies.py` from a checkout without an editable
# install; harmless when the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bankassist.config import Settings, get_settings  # noqa: E402
from bankassist.errors import BankAssistError  # noqa: E402
from bankassist.llm.factory import build_llm_client  # noqa: E402
from bankassist.logging_config import configure_logging  # noqa: E402
from bankassist.rag.embeddings import OpenAIEmbedder  # noqa: E402
from bankassist.rag.ingest import chunk_corpus, run_ingestion  # noqa: E402
from bankassist.rag.models import Chunk, PolicyDocument  # noqa: E402
from bankassist.rag.pipeline import BasicRagPipeline  # noqa: E402
from bankassist.rag.vector_store import PineconeVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk only. Makes no API call and costs nothing.",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Chunk and embed, but do not touch the vector store.",
    )
    parser.add_argument(
        "--search",
        metavar="QUESTION",
        help="Skip ingestion; run retrieval only for QUESTION against the existing index.",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Skip ingestion; run the full pipeline (retrieve + answer) for QUESTION.",
    )
    return parser.parse_args()


def print_chunk_summary(
    settings: Settings, documents: list[PolicyDocument], chunks: list[Chunk]
) -> None:
    """Print the chunk-generation summary (AC-L2-3)."""
    lengths = [len(chunk.text) for chunk in chunks]

    print(f"\nCorpus:    {settings.markdown_dir}")
    print(
        f"Chunking:  size={settings.chunk_size_chars} "
        f"window=[{settings.chunk_min_chars}, {settings.chunk_max_chars}] "
        f"overlap={settings.chunk_overlap_chars}  (characters)"
    )
    print()
    print(f"{'Document':<38} {'Category':<18} {'Chars':>8} {'Chunks':>7}")
    print("-" * 74)

    for document in documents:
        document_chunks = [c for c in chunks if c.metadata.document == document.document]
        print(
            f"{document.document:<38} {document.metadata.category:<18} "
            f"{len(document.text):>8,} {len(document_chunks):>7}"
        )

    print("-" * 74)
    print(
        f"{f'TOTAL ({len(documents)} documents)':<38} {'':<18} "
        f"{sum(len(d.text) for d in documents):>8,} {len(chunks):>7}"
    )
    print()
    print(
        f"Chunk length  min={min(lengths)}  "
        f"mean={statistics.mean(lengths):.0f}  "
        f"median={statistics.median(lengths):.0f}  "
        f"max={max(lengths)}"
    )
    print(f"Chunks above max size: {sum(1 for n in lengths if n > settings.chunk_max_chars)}")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        if args.search:
            return run_search(settings, args.search)
        if args.ask:
            return run_ask(settings, args.ask)

        # Chunk first in every mode: it is free, and it is the summary the lab
        # asks for regardless of how far the run goes.
        documents, chunks = chunk_corpus(settings)
        print_chunk_summary(settings, documents, chunks)

        if args.dry_run:
            print("\n--dry-run: stopping before embedding. No API calls were made.")
            return 0

        embedder = OpenAIEmbedder(settings)

        if args.embed_only:
            print(f"\nEmbedding {len(chunks)} chunks with {embedder.model} ...")
            started = time.perf_counter()
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            print(
                f"Embedded {len(vectors)} chunks in {time.perf_counter() - started:.1f}s  "
                f"model={embedder.model}  dimensions={len(vectors[0])}"
            )
            print("\n--embed-only: stopping before upsert. Nothing was written to Pinecone.")
            return 0

        store = PineconeVectorStore(settings)
        before = _safe_count(store)

        print(
            f"\nIngesting into Pinecone index {store.index_name!r} "
            f"namespace {store.namespace!r} with {embedder.model} ..."
        )
        result = run_ingestion(settings, embedder, store)

        print()
        print(f"Index created this run : {result.index_created}")
        print(f"Documents              : {len(result.documents)}")
        print(f"Chunks                 : {len(result.chunks)}")
        print(f"Vectors upserted       : {result.vectors_upserted}")
        print(f"Namespace count before : {before}")
        print(f"Namespace count after  : {_safe_count(store)}")
        print(f"Elapsed                : {result.elapsed_seconds:.1f}s")
        print(
            "\nRe-run this command to confirm idempotency: the namespace count "
            "must not change."
        )
        return 0
    except BankAssistError as exc:
        print(f"\n{exc.code}: {exc.message}", file=sys.stderr)
        return 1


def _build_pipeline(settings: Settings) -> tuple[BasicRagPipeline, PineconeVectorStore]:
    embedder = OpenAIEmbedder(settings)
    store = PineconeVectorStore(settings)
    llm = build_llm_client(settings)
    return BasicRagPipeline(settings, embedder, store, llm), store


def run_search(settings: Settings, question: str) -> int:
    """Retrieve for one question against the existing index (AC-L2-7 evidence)."""
    pipeline, store = _build_pipeline(settings)

    print(f'\nQuestion: "{question}"')
    print(f"Namespace count: {_safe_count(store)}  |  top_k={settings.retrieval_top_k}\n")

    results = pipeline.retrieve(question)
    if not results:
        print("No results.")
        return 0

    print(f"{'Rank':>4}  {'Score':>8}  {'Document':<32} Chunk")
    print("-" * 90)
    for rank, chunk in enumerate(results, start=1):
        preview = chunk.text[:60].replace("\n", " ")
        print(f"{rank:>4}  {chunk.score:>8.4f}  {chunk.metadata.document:<32} {preview}...")
    return 0


def run_ask(settings: Settings, question: str) -> int:
    """Full pipeline: retrieve, generate, cite (AC-L2-8/9 evidence)."""
    pipeline, store = _build_pipeline(settings)

    print(f'\nQuestion: "{question}"')
    print(f"Namespace count: {_safe_count(store)}  |  top_k={settings.retrieval_top_k}\n")

    result = pipeline.answer(question)

    print(f"Grounded: {result.grounded}\n")
    print("Answer:")
    print(result.answer)
    print()
    print("Sources:" if result.sources else "Sources: (none)")
    for source in result.sources:
        print(f"  - {source}")
    return 0


def _safe_count(store: PineconeVectorStore) -> int:
    """Namespace count, tolerating an index that does not exist yet."""
    try:
        return store.count()
    except BankAssistError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
