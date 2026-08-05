"""Evaluation runner: load the golden dataset, execute each case, score it.

The runner never talks to the graph/pipeline itself — an injected
``Executor`` does that (``evaluation.executor.GraphExecutor`` for a real run
against the live app; a stub in tests, matching the ``StubLLMClient`` pattern
used throughout the rest of the test suite). This keeps
``python -m pytest`` fully deterministic while still exercising the real
scoring logic end to end (Lab 6 requirements §21/§23).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import yaml

from evaluation.metrics import agent as agent_metrics
from evaluation.metrics import generation as generation_metrics
from evaluation.metrics import guardrail as guardrail_metrics
from evaluation.metrics import retrieval as retrieval_metrics
from evaluation.models import CaseResult, CaseScore, EvaluationReport, GoldenCase

DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_dataset.yaml"
RETRIEVAL_K = 5


class Executor(Protocol):
    """Runs one golden case against the real system and reports what happened."""

    def __call__(self, case: GoldenCase) -> CaseResult: ...


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenCase(**row) for row in raw["cases"]]


def _score_case(case: GoldenCase, result: CaseResult) -> CaseScore:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    if result.error:
        return CaseScore(
            case_id=case.case_id,
            category=case.category,
            passed=False,
            checks={"executed_without_error": False},
            notes=result.error,
            agentops_trace_id=result.agentops_trace_id,
        )

    if case.expected_sources:
        checks["retrieval_hit_at_k"] = retrieval_metrics.hit_at_k(
            result.retrieved_sources, case.expected_sources, RETRIEVAL_K
        )
        metrics["retrieval_mrr"] = retrieval_metrics.mrr(
            result.retrieved_sources, case.expected_sources
        )
        checks["citation_matches_expected"] = generation_metrics.citation_matches_expected(
            result.retrieved_sources, case.expected_sources
        )

    checks["response_exists"] = generation_metrics.response_exists(result.answer)
    if case.expected_refusal:
        checks["refusal_matches_expectation"] = generation_metrics.refusal_matches_expectation(
            result.answer, expected_refusal=True
        )
    if case.forbidden_terms:
        checks["forbidden_content_absent"] = generation_metrics.forbidden_content_absent(
            result.answer, case.forbidden_terms
        )

    if case.expected_route:
        checks["routing_correct"] = agent_metrics.routing_correct(result.route, case.expected_route)
    if case.expected_tools or case.unexpected_tools:
        checks["tools_match"] = agent_metrics.tools_match(
            result.tools_called, case.expected_tools, case.unexpected_tools
        )
    if case.approval_required is not None:
        checks["approval_gate_correct"] = agent_metrics.approval_gate_correct(
            result.approval_required, case.approval_required
        )
    if case.mutation_before_approval is not None:
        checks["mutation_never_precedes_approval"] = agent_metrics.mutation_never_precedes_approval(
            result.mutation_occurred_before_approval
        )

    if case.category in ("must_block", "must_allow"):
        checks["guardrail_verdict_correct"] = guardrail_metrics.guardrail_check_passed(case, result)

    passed = all(checks.values()) if checks else True
    return CaseScore(
        case_id=case.case_id,
        category=case.category,
        passed=passed,
        checks=checks,
        metrics=metrics,
        agentops_trace_id=result.agentops_trace_id,
    )


def _aggregate(
    cases: list[GoldenCase], results: list[CaseResult], scores: list[CaseScore]
) -> EvaluationReport:
    passed = sum(1 for s in scores if s.passed)

    routing_cases = [s for s in scores if "routing_correct" in s.checks]
    tool_cases = [s for s in scores if "tools_match" in s.checks]
    retrieval_hit_cases = [s for s in scores if "retrieval_hit_at_k" in s.checks]
    retrieval_mrr_cases = [s for s in scores if "retrieval_mrr" in s.metrics]
    citation_cases = [s for s in scores if "citation_matches_expected" in s.checks]
    pairs = list(zip(cases, results, strict=True))
    valid_latencies = [r.latency_ms for r in results if r.error is None]

    def _rate(items: list[CaseScore], key: str) -> float | None:
        return sum(1 for s in items if s.checks[key]) / len(items) if items else None

    return EvaluationReport(
        total_cases=len(scores),
        passed=passed,
        failed=len(scores) - passed,
        routing_accuracy=_rate(routing_cases, "routing_correct"),
        tool_selection_accuracy=_rate(tool_cases, "tools_match"),
        retrieval_hit_at_k=_rate(retrieval_hit_cases, "retrieval_hit_at_k"),
        retrieval_mrr=(
            sum(s.metrics["retrieval_mrr"] for s in retrieval_mrr_cases) / len(retrieval_mrr_cases)
            if retrieval_mrr_cases
            else None
        ),
        citation_accuracy=_rate(citation_cases, "citation_matches_expected"),
        attack_block_rate=guardrail_metrics.attack_block_rate(pairs),
        legitimate_allow_rate=guardrail_metrics.legitimate_allow_rate(pairs),
        average_latency_ms=(
            sum(valid_latencies) / len(valid_latencies) if valid_latencies else None
        ),
        case_scores=scores,
    )


def run_evaluation(
    executor: Executor,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    load: Callable[[Path], list[GoldenCase]] = load_dataset,
) -> EvaluationReport:
    """Run every case in the golden dataset through ``executor`` and score it.

    Never raises on a single case's failure — a case that errors is scored as
    failed with the error captured in its notes, so one bad case cannot
    silently truncate the report.
    """
    cases = load(dataset_path)
    results: list[CaseResult] = []
    for case in cases:
        try:
            result = executor(case)
        except Exception as exc:  # noqa: BLE001 - captured into the report, not swallowed silently
            result = CaseResult(case_id=case.case_id, error=f"{type(exc).__name__}: {exc}")
        results.append(result)

    scores = [_score_case(case, result) for case, result in zip(cases, results, strict=True)]
    return _aggregate(cases, results, scores)
