"""`evaluation.report.render_markdown` — only reports numbers that exist."""

from __future__ import annotations

from evaluation.models import CaseScore, EvaluationReport
from evaluation.report import render_markdown


def test_render_markdown_includes_totals_and_case_rows() -> None:
    report = EvaluationReport(
        total_cases=2,
        passed=1,
        failed=1,
        routing_accuracy=0.5,
        case_scores=[
            CaseScore(
                case_id="A", category="routing", passed=True, checks={"routing_correct": True}
            ),
            CaseScore(
                case_id="B",
                category="routing",
                passed=False,
                checks={"routing_correct": False},
                agentops_trace_id="trace-123",
            ),
        ],
    )
    markdown = render_markdown(report)
    assert "Total cases: 2" in markdown
    assert "| A | routing | PASS |" in markdown
    assert "| B | routing | FAIL |" in markdown
    assert "trace-123" in markdown
    assert "50.0%" in markdown


def test_render_markdown_shows_n_a_for_missing_metrics() -> None:
    report = EvaluationReport(total_cases=0, passed=0, failed=0)
    markdown = render_markdown(report)
    assert "n/a" in markdown
