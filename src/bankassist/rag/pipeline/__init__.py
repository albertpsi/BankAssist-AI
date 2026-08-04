"""RAG pipelines: ``basic`` (Lab 2) and ``enterprise`` (Lab 3), selectable by mode.

Re-exports both so ``from bankassist.rag.pipeline import BasicRagPipeline`` keeps
working unchanged after the Lab 3 package move (FR-L3-1's mechanical-move
requirement) — no import site outside this package needed to change.
"""

from __future__ import annotations

from bankassist.rag.pipeline.basic_pipeline import BasicRagPipeline
from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline

__all__ = ["BasicRagPipeline", "EnterpriseRagPipeline"]
