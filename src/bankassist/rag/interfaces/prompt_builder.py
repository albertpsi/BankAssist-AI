"""The ``PromptBuilder`` seam (FR-L3-9)."""

from __future__ import annotations

from typing import Protocol

from bankassist.rag.models.retrieval_context import PromptBuildRequest, PromptContext


class PromptBuilderProtocol(Protocol):
    def execute(self, request: PromptBuildRequest) -> PromptContext:
        """Assemble the grounded prompt from a rerank result."""
        ...
