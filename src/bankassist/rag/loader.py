"""Read the policy corpus from disk.

The corpus is three parallel directories under ``data/policies``::

    markdown/<stem>.md    the text that gets ingested
    metadata/<stem>.json  the sidecar carrying title, category, source
    pdf/<stem>.pdf        the original document, provenance only — never read

Pairing is by file stem. A markdown file with no sidecar is an error rather than
a document with blank metadata: an uncited chunk is worse than a failed ingest.
"""

from __future__ import annotations

import json
from pathlib import Path

from bankassist.errors import IngestionError
from bankassist.logging_config import get_logger
from bankassist.rag.models import DocumentMetadata, PolicyDocument

logger = get_logger(__name__)

REQUIRED_METADATA_KEYS = ("title", "category", "source")


def normalize_text(raw: str) -> str:
    """Whitespace-only cleanup, applied before chunking.

    The corpus is PDF-extracted, so it carries page banners, flattened tables,
    and stray one-line headings. Those are deliberately left alone — repairing
    them is layout-aware parsing, which Lab 2 excludes. Only line endings,
    trailing spaces, and runs of blank lines are touched, because those change
    chunk boundaries without changing meaning.

    Blank runs collapse to a *single* blank line: one is all the chunker needs to
    see a paragraph boundary, and every extra one spends chunk budget on nothing.
    """
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            collapsed.append(line)
            continue
        blank_run += 1
        if blank_run == 1:
            collapsed.append(line)

    return "\n".join(collapsed).strip()


def load_corpus(markdown_dir: Path, metadata_dir: Path) -> list[PolicyDocument]:
    """Load every document in the corpus, sorted by file name.

    Sorted so that ingestion order — and therefore the log output and any
    ordering-sensitive assertion — is reproducible across machines.

    Raises:
        IngestionError: a directory is missing, a sidecar is absent or
            unparseable, or a required metadata key is missing or blank.
    """
    if not markdown_dir.is_dir():
        raise IngestionError(
            f"Policy markdown directory not found: {markdown_dir}",
            details={"path": str(markdown_dir)},
        )
    if not metadata_dir.is_dir():
        raise IngestionError(
            f"Policy metadata directory not found: {metadata_dir}",
            details={"path": str(metadata_dir)},
        )

    paths = sorted(markdown_dir.glob("*.md"))
    if not paths:
        raise IngestionError(
            f"No markdown documents found in {markdown_dir}",
            details={"path": str(markdown_dir)},
        )

    documents = [load_document(path, metadata_dir) for path in paths]

    logger.info(
        "corpus loaded",
        extra={
            "markdown_dir": str(markdown_dir),
            "document_count": len(documents),
            "total_chars": sum(len(doc.text) for doc in documents),
        },
    )
    return documents


def load_document(markdown_path: Path, metadata_dir: Path) -> PolicyDocument:
    """Load one markdown file and its sidecar."""
    metadata = _load_metadata(markdown_path, metadata_dir)
    text = normalize_text(markdown_path.read_text(encoding="utf-8"))

    if not text:
        raise IngestionError(
            f"Policy document is empty: {markdown_path.name}",
            details={"document": markdown_path.name},
        )

    return PolicyDocument(metadata=metadata, text=text)


def _load_metadata(markdown_path: Path, metadata_dir: Path) -> DocumentMetadata:
    """Read the sidecar for a markdown file, matched by stem."""
    sidecar = metadata_dir / f"{markdown_path.stem}.json"
    if not sidecar.is_file():
        raise IngestionError(
            f"No metadata sidecar for {markdown_path.name}: expected {sidecar.name}",
            details={"document": markdown_path.name, "expected": sidecar.name},
        )

    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(
            f"Metadata sidecar {sidecar.name} is not valid JSON: {exc.msg} (line {exc.lineno})",
            details={"document": markdown_path.name, "sidecar": sidecar.name},
        ) from exc

    if not isinstance(payload, dict):
        raise IngestionError(
            f"Metadata sidecar {sidecar.name} must contain a JSON object.",
            details={"document": markdown_path.name, "sidecar": sidecar.name},
        )

    missing = [key for key in REQUIRED_METADATA_KEYS if not str(payload.get(key) or "").strip()]
    if missing:
        raise IngestionError(
            f"Metadata sidecar {sidecar.name} is missing required key(s): {', '.join(missing)}",
            details={"document": markdown_path.name, "missing": missing},
        )

    # Every other sidecar key — id, url, document_type, version, effective_date,
    # download_date, language, origin — is deliberately dropped. Those become
    # metadata filters in Lab 3; storing them now would invite filtering now.
    return DocumentMetadata(
        document=markdown_path.name,
        title=str(payload["title"]).strip(),
        category=str(payload["category"]).strip(),
        source=str(payload["source"]).strip(),
    )
