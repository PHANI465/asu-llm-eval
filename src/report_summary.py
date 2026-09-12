# =============================================================================
# src/report_summary.py
# ASU LLM Evaluation — CI summary of results/latest_report.json
#
# Prints a plain-text summary to the Actions log and, when GITHUB_STEP_SUMMARY
# is set (always true inside GitHub Actions), appends a Markdown table to the
# run's summary page.
#
# Usage: python src/report_summary.py [path/to/latest_report.json]
# =============================================================================

import json
import os
import sys

from project_config import RESULTS_DIR

GATE_LABELS = {
    "hallucination_rate": "Hallucination rate",
    "answer_relevancy":   "Answer relevancy",
    "faithfulness":       "Faithfulness",
    "context_precision":  "Context precision",
    "latency_p95":        "Latency p95",
    "cost_per_query":     "Cost per query",
    "error_rate":         "Error rate",
}


def format_gate_value(gate_name: str, value) -> str:
    if value is None:
        return "N/A"
    if gate_name in ("hallucination_rate", "error_rate"):
        return f"{value:.1%}"
    if gate_name == "latency_p95":
        return f"{value:.2f}s"
    if gate_name == "cost_per_query":
        return f"${value:.4f}"
    return f"{value:.4f}"


def build_text(report: dict) -> str:
    overall = report.get("overall_result", "UNKNOWN")
    lines = ["=" * 44, "EVALUATION SUMMARY", "=" * 44]

    if overall == "ERROR":
        lines += [
            "  STATUS  : ERROR (evaluation did not complete)",
            f"  Commit  : {report.get('commit_id')}",
            f"  Error   : {report.get('pipeline_error', 'unknown error')}",
        ]
        if report.get("error_hint"):
            lines.append(f"  Hint    : {report['error_hint']}")
        return "\n".join(lines + ["=" * 44])

    models = report.get("models") or {}
    mode   = "TEST MODE" if report.get("test_mode") else "FULL RUN"
    lines += [
        f"  Questions : {report.get('total_questions')} of {report.get('dataset_size')} ({mode})",
        f"  Commit    : {report.get('commit_id')}",
        f"  Models    : {models.get('answer', '?')} answers, {models.get('judge', '?')} judge",
        f"  Run cost  : ${report.get('total_cost_usd') or 0:.4f}",
        "",
    ]
    for gate_name, gate in (report.get("gate_results", {}).get("gates") or {}).items():
        verdict = "PASS" if gate.get("passed") else "FAIL"
        lines.append(f"  {GATE_LABELS.get(gate_name, gate_name):<20}: {verdict}  "
                     f"{format_gate_value(gate_name, gate.get('value'))}")
    lines += ["", f"  OVERALL: {overall}"]
    if report.get("failed_gates"):
        lines.append(f"  FAILED : {', '.join(report['failed_gates'])}")
    return "\n".join(lines + ["=" * 44])


def build_markdown(report: dict) -> str:
    overall = report.get("overall_result", "UNKNOWN")
    icon    = {"PASS": "✅", "FAIL": "❌"}.get(overall, "⚠️")

    if overall == "ERROR":
        md = [
            f"## {icon} Evaluation ERROR — the pipeline could not run",
            "",
            f"**Error:** `{report.get('pipeline_error', 'unknown error')}`",
        ]
        if report.get("error_hint"):
            md += ["", f"**Hint:** {report['error_hint']}"]
        return "\n".join(md) + "\n"

    models = report.get("models") or {}
    mode   = "test mode" if report.get("test_mode") else "full run"
    md = [
        f"## {icon} Evaluation {overall}",
        "",
        f"**{report.get('total_questions')} of {report.get('dataset_size')} questions** ({mode}) · "
        f"commit `{report.get('commit_id')}` · {models.get('answer', '?')} answers, "
        f"{models.get('judge', '?')} judge · run cost ${report.get('total_cost_usd') or 0:.4f}",
        "",
        "| Gate | Value | Threshold | Status |",
        "|---|---|---|---|",
    ]
    for gate_name, gate in (report.get("gate_results", {}).get("gates") or {}).items():
        threshold = gate.get("threshold")
        symbol    = "≤" if gate.get("direction") == "max" else "≥"
        md.append(
            f"| {GATE_LABELS.get(gate_name, gate_name)} "
            f"| {format_gate_value(gate_name, gate.get('value'))} "
            f"| {symbol} {format_gate_value(gate_name, threshold) if threshold is not None else '—'} "
            f"| {'✅ pass' if gate.get('passed') else '❌ fail'} |"
        )

    failures = report.get("sample_failures") or []
    if failures:
        md += ["", f"**{report.get('failed_question_count', len(failures))} failed question(s)** — worst {len(failures)}:", ""]
        for f in failures:
            detail = f["error"] if f.get("error") else f"faithfulness {f.get('faithfulness')}"
            md.append(f"- {f.get('question')} — _{detail}_")
    return "\n".join(md) + "\n"


def main(path: str) -> int:
    if not os.path.exists(path):
        print("No report file found.")
        print("The pipeline crashed before it could write a report — see the evaluation step log above.")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(build_text(report))

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(build_markdown(report))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "latest_report.json")))
