"""Character-based chunking.

The Lab 2 brief fixes the shape: ~700–900 characters with 100–150 of overlap, and
no semantic or layout-aware splitting. So this is a fixed-size window with a
*boundary preference* — inside the accepted size window it would rather cut at a
paragraph break than a sentence end, and at a sentence end than mid-word — but it
never looks at what the text means.

Pure and deterministic: no I/O, no model call, same input always the same output.
"""

from __future__ import annotations

import re

from bankassist.rag.models import Chunk, PolicyDocument

# End a chunk *before* the blank line, so no chunk carries a trailing separator.
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
# A sentence terminator followed by whitespace, allowing a closing quote or bracket.
# Ends *after* the punctuation, so the sentence stays intact.
_SENTENCE = re.compile(r"[.!?][)\"'\]]*(?=\s)")
_WHITESPACE = re.compile(r"\s")


def chunk_document(
    document: PolicyDocument,
    *,
    size: int,
    min_size: int,
    max_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split one document into overlapping chunks, metadata attached."""
    spans = chunk_spans(
        document.text, size=size, min_size=min_size, max_size=max_size, overlap=overlap
    )
    return [
        Chunk(
            metadata=document.metadata,
            text=document.text[start:end],
            chunk_index=index,
            char_start=start,
            char_end=end,
        )
        for index, (start, end) in enumerate(spans)
    ]


def chunk_spans(
    text: str,
    *,
    size: int,
    min_size: int,
    max_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` pairs indexing into ``text``.

    Spans rather than strings so the caller can always prove where a chunk came
    from: ``text[start:end]`` is the chunk, exactly.

    The final span of a document is whatever is left and may be shorter than
    ``min_size``; every other span lands within ``[min_size, max_size]``.
    """
    length = len(text)
    spans: list[tuple[int, int]] = []
    start = 0

    while start < length:
        start = _skip_whitespace(text, start, length)
        if start >= length:
            break

        # Everything that remains fits: take it and stop, rather than emitting a
        # tiny orphan chunk after an otherwise full-size one.
        if length - start <= max_size:
            cut = length
        else:
            cut = _find_break(text, start, size, min_size, max_size)

        end = _trim_trailing_whitespace(text, start, cut)
        if end > start:
            spans.append((start, end))

        if cut >= length:
            break

        # `overlap < min_size` is enforced by settings validation, so this always
        # advances. The guard covers a direct caller that bypassed settings.
        start = max(cut - overlap, start + 1)

    return spans


def _find_break(text: str, start: int, size: int, min_size: int, max_size: int) -> int:
    """Choose where to cut, given the chunk must end within the size window.

    Tries paragraph breaks, then sentence ends, then any whitespace; within each
    class it takes the candidate nearest the target size, so chunks stay evenly
    sized instead of drifting to one end of the window. Falls back to a hard cut
    at ``max_size`` for text with no break at all — a long table or an unbroken
    run of characters, both of which this corpus contains.
    """
    window_lo = start + min_size
    window_hi = start + max_size
    target = start + size
    window = text[window_lo:window_hi]

    preferences = ((_PARAGRAPH, False), (_SENTENCE, True), (_WHITESPACE, False))
    for pattern, ends_after_match in preferences:
        candidates = [
            window_lo + (match.end() if ends_after_match else match.start())
            for match in pattern.finditer(window)
        ]
        # A candidate at `start` would produce an empty chunk and stall progress.
        candidates = [candidate for candidate in candidates if candidate > start]
        if candidates:
            return min(candidates, key=lambda candidate: abs(candidate - target))

    return window_hi


def _skip_whitespace(text: str, index: int, length: int) -> int:
    """Advance past whitespace so a chunk never opens on a blank line."""
    while index < length and text[index].isspace():
        index += 1
    return index


def _trim_trailing_whitespace(text: str, start: int, end: int) -> int:
    """Pull the end back off any trailing whitespace, keeping the span exact."""
    while end > start and text[end - 1].isspace():
        end -= 1
    return end
