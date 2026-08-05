"""`evaluation.runner` scored against a fully deterministic stub executor —
no network, no real graph, matching the `StubLLMClient` pattern used
throughout the rest of the test suite (Lab 6 §21)."""

from __future__ import annotations

from pathlib import Path

import yaml
from evaluation.models import CaseResult, GoldenCase
from evaluation.runner import load_dataset, run_evaluation

_DATASET_PATH = Path(__file__).parent.parent.parent / "evaluation" / "golden_dataset.yaml"


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump({"cases": cases}), encoding="utf-8")
    return path


def test_golden_dataset_loads_and_has_20_to_25_cases() -> None:
    cases = load_dataset(_DATASET_PATH)
    assert 20 <= len(cases) <= 25
    categories = {c.category for c in cases}
    assert {
        "policy_rag",
        "banking_agent",
        "dispute_agent",
        "routing",
        "multi_turn",
        "must_block",
        "must_allow",
    }.issubset(categories)


def test_run_evaluation_scores_a_passing_rag_case(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            {
                "case_id": "RAG-T1",
                "category": "policy_rag",
                "query": "q",
                "expected_sources": ["doc.pdf"],
            }
        ],
    )

    def executor(case: GoldenCase) -> CaseResult:
        return CaseResult(
            case_id=case.case_id,
            answer="Here is the answer.",
            retrieved_sources=["doc.pdf", "other.pdf"],
        )

    report = run_evaluation(executor, dataset_path=dataset)
    assert report.total_cases == 1
    assert report.passed == 1
    assert report.retrieval_hit_at_k == 1.0
    assert report.retrieval_mrr == 1.0


def test_run_evaluation_scores_a_failing_routing_case(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            {
                "case_id": "ROUTE-T1",
                "category": "routing",
                "query": "q",
                "expected_route": "BANKING",
            }
        ],
    )

    def executor(case: GoldenCase) -> CaseResult:
        return CaseResult(case_id=case.case_id, answer="ok", route="POLICY")

    report = run_evaluation(executor, dataset_path=dataset)
    assert report.passed == 0
    assert report.failed == 1
    assert report.routing_accuracy == 0.0
    assert report.case_scores[0].checks["routing_correct"] is False


def test_run_evaluation_computes_guardrail_rates(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            {"case_id": "B1", "category": "must_block", "query": "attack", "must_block": True},
            {"case_id": "A1", "category": "must_allow", "query": "normal", "must_block": False},
        ],
    )

    def executor(case: GoldenCase) -> CaseResult:
        return CaseResult(case_id=case.case_id, answer="ok", blocked=case.must_block)

    report = run_evaluation(executor, dataset_path=dataset)
    assert report.attack_block_rate == 1.0
    assert report.legitimate_allow_rate == 1.0
    assert report.passed == 2


def test_run_evaluation_captures_executor_errors_without_aborting(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            {"case_id": "E1", "category": "routing", "query": "q"},
            {"case_id": "E2", "category": "routing", "query": "q"},
        ],
    )

    def executor(case: GoldenCase) -> CaseResult:
        if case.case_id == "E1":
            raise RuntimeError("boom")
        return CaseResult(case_id=case.case_id, answer="ok")

    report = run_evaluation(executor, dataset_path=dataset)
    assert report.total_cases == 2
    assert report.case_scores[0].passed is False
    assert "boom" in report.case_scores[0].notes
    assert report.case_scores[1].passed is True


def test_mutation_before_approval_case_fails_when_violated(tmp_path: Path) -> None:
    dataset = _write_dataset(
        tmp_path,
        [
            {
                "case_id": "DISP-T1",
                "category": "dispute_agent",
                "query": "q",
                "mutation_before_approval": False,
            }
        ],
    )

    def executor(case: GoldenCase) -> CaseResult:
        return CaseResult(case_id=case.case_id, answer="ok", mutation_occurred_before_approval=True)

    report = run_evaluation(executor, dataset_path=dataset)
    assert report.passed == 0
    assert report.case_scores[0].checks["mutation_never_precedes_approval"] is False
