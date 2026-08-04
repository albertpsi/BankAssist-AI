"""Policy Agent (Lab 4, §3). Thin wrapper over Lab 3's ``EnterpriseRagPipeline``.

No retrieval, classification, or rerank logic is duplicated here — every stage stays
exactly where Lab 3 left it.
"""

from __future__ import annotations

from dataclasses import dataclass

from bankassist.rag.pipeline.enterprise_pipeline import EnterpriseRagPipeline


@dataclass(frozen=True)
class PolicyAnswer:
    answer: str
    sources: list[str]


def answer_policy_question(pipeline: EnterpriseRagPipeline, question: str) -> PolicyAnswer:
    result = pipeline.answer(question)
    return PolicyAnswer(answer=result.generated_answer, sources=result.citations)
