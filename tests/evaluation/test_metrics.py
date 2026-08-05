"""Deterministic unit tests for evaluation/metrics/*.py — no LLM, no network."""

from __future__ import annotations

import pytest
from evaluation.metrics import agent as agent_metrics
from evaluation.metrics import generation as generation_metrics
from evaluation.metrics import guardrail as guardrail_metrics
from evaluation.metrics import retrieval as retrieval_metrics
from evaluation.models import CaseResult, GoldenCase

# --- retrieval ---


def test_hit_at_k_true_when_expected_source_in_window() -> None:
    assert retrieval_metrics.hit_at_k(["a", "b", "c"], ["c"], k=3) is True


def test_hit_at_k_false_when_expected_source_outside_window() -> None:
    assert retrieval_metrics.hit_at_k(["a", "b", "c", "d"], ["d"], k=2) is False


def test_hit_at_k_requires_expected_sources() -> None:
    with pytest.raises(ValueError, match="expected"):
        retrieval_metrics.hit_at_k(["a"], [], k=3)


def test_recall_at_k_partial_match() -> None:
    assert retrieval_metrics.recall_at_k(["a", "b"], ["a", "c"], k=2) == pytest.approx(0.5)


def test_mrr_first_hit_at_rank_two() -> None:
    assert retrieval_metrics.mrr(["a", "b", "c"], ["b"]) == pytest.approx(0.5)


def test_mrr_no_hit_is_zero() -> None:
    assert retrieval_metrics.mrr(["a", "b"], ["z"]) == 0.0


# --- generation ---


def test_response_exists_false_on_blank() -> None:
    assert generation_metrics.response_exists("   ") is False


def test_refusal_matches_expectation_true_positive() -> None:
    answer = "I can't help with that request — and I can't provide personalized investment advice."
    assert generation_metrics.refusal_matches_expectation(answer, expected_refusal=True) is True


def test_refusal_matches_expectation_false_when_not_a_refusal() -> None:
    answer = "Here is a summary of your accounts."
    assert generation_metrics.refusal_matches_expectation(answer, expected_refusal=True) is False


def test_citation_matches_expected_requires_overlap() -> None:
    assert generation_metrics.citation_matches_expected(["a.pdf"], ["b.pdf"]) is False
    assert generation_metrics.citation_matches_expected(["a.pdf"], ["a.pdf"]) is True


def test_citation_matches_expected_vacuously_true_with_no_expectation() -> None:
    assert generation_metrics.citation_matches_expected([], []) is True


def test_forbidden_content_absent() -> None:
    assert generation_metrics.forbidden_content_absent("your balance is fine", ["4111"]) is True
    assert generation_metrics.forbidden_content_absent("card 4111111111111111", ["4111"]) is False


# --- agent ---


def test_routing_correct_case_insensitive() -> None:
    assert agent_metrics.routing_correct("banking", "BANKING") is True


def test_routing_correct_vacuously_true_with_no_expectation() -> None:
    assert agent_metrics.routing_correct(None, None) is True


def test_tools_match_requires_all_expected_and_none_unexpected() -> None:
    assert agent_metrics.tools_match(
        ["get_recent_transactions"], ["get_recent_transactions"], ["create_dispute"]
    )
    assert not agent_metrics.tools_match(["create_dispute"], [], ["create_dispute"])
    assert not agent_metrics.tools_match([], ["get_recent_transactions"], [])


def test_approval_gate_correct() -> None:
    assert agent_metrics.approval_gate_correct(True, True) is True
    assert agent_metrics.approval_gate_correct(False, True) is False
    assert agent_metrics.approval_gate_correct(True, None) is True


def test_mutation_never_precedes_approval() -> None:
    assert agent_metrics.mutation_never_precedes_approval(False) is True
    assert agent_metrics.mutation_never_precedes_approval(True) is False


# --- guardrail ---


def _case(must_block: bool, category: str = "must_block") -> GoldenCase:
    return GoldenCase(case_id="X", category=category, query="q", must_block=must_block)


def test_guardrail_check_passed_block_case() -> None:
    case = _case(must_block=True)
    assert guardrail_metrics.guardrail_check_passed(case, CaseResult(case_id="X", blocked=True))
    assert not guardrail_metrics.guardrail_check_passed(
        case, CaseResult(case_id="X", blocked=False)
    )


def test_guardrail_check_passed_allow_case() -> None:
    case = _case(must_block=False, category="must_allow")
    assert guardrail_metrics.guardrail_check_passed(case, CaseResult(case_id="X", blocked=False))
    assert not guardrail_metrics.guardrail_check_passed(
        case, CaseResult(case_id="X", blocked=True)
    )


def test_attack_block_rate_and_legitimate_allow_rate() -> None:
    pairs = [
        (_case(True), CaseResult(case_id="B1", blocked=True)),
        (_case(True), CaseResult(case_id="B2", blocked=False)),
        (_case(False, "must_allow"), CaseResult(case_id="A1", blocked=False)),
    ]
    assert guardrail_metrics.attack_block_rate(pairs) == pytest.approx(0.5)
    assert guardrail_metrics.legitimate_allow_rate(pairs) == pytest.approx(1.0)


def test_rates_are_none_when_no_applicable_cases() -> None:
    assert guardrail_metrics.attack_block_rate([]) is None
    assert guardrail_metrics.legitimate_allow_rate([]) is None
