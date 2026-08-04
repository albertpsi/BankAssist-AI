"""Query classification output (FR-L3-3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClassificationLabel = Literal[
    "Policy", "FAQ", "Procedure", "Eligibility", "Definition", "Comparison", "Unknown"
]


class ClassificationResult(BaseModel):
    """The route a question was classified into, and how sure the model was."""

    label: ClassificationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
