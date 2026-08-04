"""Prompt construction for the enterprise pipeline (FR-L3-9)."""

from __future__ import annotations

from bankassist.logging_config import get_logger
from bankassist.rag.models.retrieval_context import PromptBuildRequest, PromptContext
from bankassist.rag.prompts import build_enterprise_messages

logger = get_logger(__name__)

# A rough, dependency-free estimate (design decision, lab-03 spec §5.5): not
# exact token accounting, which is out of scope without `tiktoken`.
_CHARS_PER_TOKEN_ESTIMATE = 4


class PromptBuilder:
    def execute(self, request: PromptBuildRequest) -> PromptContext:
        entries = request.reranked.entries
        messages = build_enterprise_messages(
            request.original_question, request.rewritten_question, entries
        )

        total_chars = sum(len(message.content) for message in messages)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN_ESTIMATE

        context = PromptContext(
            messages=messages,
            chunk_count=len(entries),
            estimated_tokens=estimated_tokens,
        )
        logger.info(
            "prompt construction",
            extra={
                "chunk_count": context.chunk_count,
                "estimated_tokens": context.estimated_tokens,
            },
        )
        return context
