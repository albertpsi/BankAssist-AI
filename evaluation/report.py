"""Render an `EvaluationReport` as the compact Markdown report (Lab 6 §18).

Only reports numbers the runner actually computed — a metric with no
applicable cases renders as "n/a", never a fabricated 0 or 100%.
"""

from __future__ import annotations

from evaluation.models import EvaluationReport


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else str(value)


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# BankAssist AI — Lab 6 Evaluation Report",
        "",
        f"- Total cases: {report.total_cases}",
        f"- Passed: {report.passed}",
        f"- Failed: {report.failed}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Routing accuracy | {_pct(report.routing_accuracy)} |",
        f"| Tool selection accuracy | {_pct(report.tool_selection_accuracy)} |",
        f"| Retrieval Hit@K | {_pct(report.retrieval_hit_at_k)} |",
        f"| Retrieval MRR | {_num(report.retrieval_mrr)} |",
        f"| Citation accuracy | {_pct(report.citation_accuracy)} |",
        f"| Attack block rate | {_pct(report.attack_block_rate)} |",
        f"| Legitimate allow rate | {_pct(report.legitimate_allow_rate)} |",
        f"| Average latency | {_ms(report.average_latency_ms)} |",
        "",
        "## Case results",
        "",
        "| Case | Category | Result | AgentOps trace | Notes |",
        "|---|---|---|---|---|",
    ]
    for score in report.case_scores:
        result = "PASS" if score.passed else "FAIL"
        failed_checks = ", ".join(k for k, v in score.checks.items() if not v)
        trace = score.agentops_trace_id or "—"
        notes = score.notes or failed_checks
        lines.append(f"| {score.case_id} | {score.category} | {result} | {trace} | {notes} |")

    return "\n".join(lines) + "\n"
