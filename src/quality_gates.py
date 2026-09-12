# =============================================================================
# src/quality_gates.py
# ASU LLM Evaluation — Quality Gate Engine
#
# Responsibilities:
#   1. compute_hallucination_rate(results) -> float | None
#         Share of scored answers whose faithfulness falls below
#         evaluation.hallucination_faithfulness_threshold (default 0.5)
#
#   2. compute_aggregate_metrics(results) -> dict
#         Aggregates all per-question scores into a single metrics dict
#         suitable for passing straight to check_gates().
#
#   3. check_gates(metrics) -> dict
#         Reads thresholds from config.yaml, evaluates each metric,
#         and returns a structured PASS/FAIL report.
#
#   4. print_gate_report(gate_results)
#         Prints a human-readable gate report to stdout.
# =============================================================================

import numpy as np

from project_config import load_config

DEFAULT_HALLUCINATION_THRESHOLD = 0.5

# -----------------------------------------------------------------------------
# Gate definitions:
#   (metric_key, threshold_key, direction, gate_name)
#   direction: "max" = value must be AT OR BELOW threshold (lower is better)
#              "min" = value must be AT OR ABOVE threshold (higher is better)
# -----------------------------------------------------------------------------
GATE_DEFINITIONS = [
    ("hallucination_rate",  "hallucination_rate_max",  "max", "hallucination_rate"),
    ("answer_relevancy",    "answer_relevancy_min",    "min", "answer_relevancy"),
    ("faithfulness",        "faithfulness_min",        "min", "faithfulness"),
    ("context_precision",   "context_precision_min",   "min", "context_precision"),
    ("latency_p95_seconds", "latency_p95_max_seconds", "max", "latency_p95"),
    ("cost_per_query_usd",  "cost_per_query_max_usd",  "max", "cost_per_query"),
    ("error_rate",          "error_rate_max",          "max", "error_rate"),
]


def hallucination_threshold() -> float:
    """Faithfulness below this marks an answer as hallucinated (config.yaml → evaluation)."""
    return load_config()["evaluation"].get(
        "hallucination_faithfulness_threshold", DEFAULT_HALLUCINATION_THRESHOLD
    )


# -----------------------------------------------------------------------------
# 1. Hallucination rate (derived metric)
# -----------------------------------------------------------------------------

def compute_hallucination_rate(results: list, threshold: float | None = None) -> float | None:
    """
    Share of scored answers that hallucinated.

    Definition
    ----------
    hallucination_rate = count(faithfulness < threshold) / count(scored answers)

    This counts *answers* that are mostly unsupported by the retrieved
    context, so it complements the faithfulness gate (a mean over claims)
    instead of duplicating it — a run can have a high mean faithfulness and
    still contain a few fabricated answers.

    Returns None when no answer was scored, so the gate fails instead of
    optimistically reporting 0% for a run that produced no evidence.
    """
    if threshold is None:
        threshold = hallucination_threshold()

    scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    if not scores:
        return None

    return round(sum(score < threshold for score in scores) / len(scores), 4)


# -----------------------------------------------------------------------------
# 2. Aggregate metrics across the full result set
# -----------------------------------------------------------------------------

def compute_aggregate_metrics(results: list, hallucination_faithfulness_threshold: float | None = None) -> dict:
    """
    Compute aggregate metrics from a list of per-question result dicts.

    Parameters
    ----------
    results : list of dicts combining rag_pipeline.get_answer() output with
              evaluator scores. Used keys:
                faithfulness, answer_relevancy, context_precision,
                latency_seconds, cost_usd, error
              None scores are skipped per metric. Results with an "error"
              (the RAG call failed) count toward error_rate only — their
              near-instant failure latency and zero cost would otherwise make
              latency and cost look better than they are.

    Returns
    -------
    dict with keys:
        hallucination_rate    float  — share of answers with low faithfulness
        answer_relevancy      float  — mean answer relevancy
        faithfulness          float  — mean faithfulness
        context_precision     float  — mean context precision
        latency_p95_seconds   float  — 95th-percentile latency of successful calls
        cost_per_query_usd    float  — mean answer-model cost of successful calls
        error_rate            float  — share of questions whose RAG call failed
    Any metric without data is None (its gate then fails).
    """
    if not results:
        raise ValueError("results list is empty — nothing to aggregate.")

    successful = [r for r in results if not r.get("error")]

    def _mean(key: str, rows: list, digits: int = 4) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), digits) if values else None

    latencies = [r["latency_seconds"] for r in successful if r.get("latency_seconds") is not None]

    return {
        "hallucination_rate":  compute_hallucination_rate(results, hallucination_faithfulness_threshold),
        "answer_relevancy":    _mean("answer_relevancy", results),
        "faithfulness":        _mean("faithfulness", results),
        "context_precision":   _mean("context_precision", results),
        "latency_p95_seconds": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
        "cost_per_query_usd":  _mean("cost_usd", successful, digits=6),
        "error_rate":          round((len(results) - len(successful)) / len(results), 4),
    }


# -----------------------------------------------------------------------------
# 3. Gate evaluation
# -----------------------------------------------------------------------------

def check_gates(metrics: dict, thresholds: dict | None = None) -> dict:
    """
    Evaluate each aggregate metric against its configured threshold.

    Parameters
    ----------
    metrics    : dict from compute_aggregate_metrics()
    thresholds : optional override; defaults to config.yaml → quality_gates

    Returns
    -------
    dict:
        overall       : "PASS" or "FAIL"
        gates         : dict of per-gate result dicts
        failed_gates  : list of gate names that failed
        passed_gates  : list of gate names that passed
    """
    if thresholds is None:
        thresholds = load_config()["quality_gates"]

    gates        = {}
    passed_gates = []
    failed_gates = []

    for metric_key, threshold_key, direction, gate_name in GATE_DEFINITIONS:
        value     = metrics.get(metric_key)
        threshold = thresholds.get(threshold_key)

        if threshold is None:
            gates[gate_name] = {
                "value":     value,
                "threshold": None,
                "direction": direction,
                "passed":    True,   # no threshold configured → do not block
                "message":   f"SKIP: no threshold configured for '{threshold_key}'",
            }
            passed_gates.append(gate_name)
            continue

        # A missing value means nothing could be measured — never a pass
        if value is None:
            gates[gate_name] = {
                "value":     None,
                "threshold": threshold,
                "direction": direction,
                "passed":    False,
                "message":   f"FAIL: no value for '{metric_key}' (nothing was scored)",
            }
            failed_gates.append(gate_name)
            continue

        if direction == "max":
            passed  = value <= threshold
            op_word = "within max" if passed else "exceeds max"
        else:  # direction == "min"
            passed  = value >= threshold
            op_word = "meets min" if passed else "below min"

        gates[gate_name] = {
            "value":     value,
            "threshold": threshold,
            "direction": direction,
            "passed":    passed,
            "message":   f"{'PASS' if passed else 'FAIL'}: {value} {op_word} {threshold}",
        }
        (passed_gates if passed else failed_gates).append(gate_name)

    return {
        "overall":      "PASS" if not failed_gates else "FAIL",
        "gates":        gates,
        "failed_gates": failed_gates,
        "passed_gates": passed_gates,
    }


# -----------------------------------------------------------------------------
# 4. Human-readable report printer
# -----------------------------------------------------------------------------

_REPORT_FORMAT = {
    # gate_name: (value formatter, threshold formatter)
    "hallucination_rate": (lambda v: f"{v:.4f}",  lambda t: f"{t}"),
    "answer_relevancy":   (lambda v: f"{v:.4f}",  lambda t: f"{t}"),
    "faithfulness":       (lambda v: f"{v:.4f}",  lambda t: f"{t}"),
    "context_precision":  (lambda v: f"{v:.4f}",  lambda t: f"{t}"),
    "latency_p95":        (lambda v: f"{v}s",     lambda t: f"{t}s"),
    "cost_per_query":     (lambda v: f"${v:.4f}", lambda t: f"${t}"),
    "error_rate":         (lambda v: f"{v:.1%}",  lambda t: f"{t:.0%}"),
}


def print_gate_report(gate_results: dict) -> None:
    """
    Print a formatted quality-gate report to stdout.

    Parameters
    ----------
    gate_results : dict returned by check_gates()
    """
    DIVIDER = "=" * 44

    print(DIVIDER)
    print("QUALITY GATE REPORT")
    print(DIVIDER)

    for gate_name, info in gate_results["gates"].items():
        verdict   = "PASS" if info["passed"] else "FAIL"
        value     = info["value"]
        threshold = info["threshold"]

        if threshold is None:
            detail = "(skipped: no threshold configured)"
        elif value is None:
            detail = "(no value: nothing was scored)"
        else:
            fmt_v, fmt_t = _REPORT_FORMAT.get(gate_name, (str, str))
            sym = "<=" if info.get("direction") == "max" else ">="
            detail = f"({fmt_v(value)} {sym} {fmt_t(threshold)})"

        print(f"  {gate_name:<20}: {verdict:<4}  {detail}")

    print(DIVIDER)

    overall      = gate_results["overall"]
    failed_gates = gate_results["failed_gates"]

    print(f"OVERALL RESULT: {overall}")
    if overall == "PASS":
        print(f"All {len(gate_results['passed_gates'])} gates passed. Safe to deploy.")
    else:
        print(f"{len(failed_gates)} gate(s) failed. Deployment blocked.")
        print(f"Failed gates: {', '.join(failed_gates)}")

    print(DIVIDER)


# -----------------------------------------------------------------------------
# 5. Quick-test entry point (offline — no API calls)
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 44)
    print("  Quality Gates - Scenario Tests")
    print("=" * 44)

    base_metrics = {
        "hallucination_rate":  0.0,
        "answer_relevancy":    0.82,
        "faithfulness":        0.91,
        "context_precision":   0.75,
        "latency_p95_seconds": 2.1,
        "cost_per_query_usd":  0.005,
        "error_rate":          0.0,
    }

    print("\n--- Scenario 1: All gates should PASS ---\n")
    print_gate_report(check_gates(base_metrics))

    print("\n--- Scenario 2: Two gates should FAIL ---\n")
    print_gate_report(check_gates({**base_metrics, "hallucination_rate": 0.2, "answer_relevancy": 0.60}))

    print("\n--- Scenario 3: Every RAG call failed (e.g. OpenAI quota exhausted) ---\n")
    outage = [
        {"question": f"Q{i}", "answer": "", "error": "RateLimitError: insufficient_quota",
         "faithfulness": None, "answer_relevancy": None, "context_precision": None,
         "latency_seconds": 1.4, "token_usage": 0, "cost_usd": None}
        for i in range(10)
    ]
    print_gate_report(check_gates(compute_aggregate_metrics(outage)))

    print("\n--- Scenario 4: compute_aggregate_metrics with mock data ---\n")
    mock_results = [
        {"faithfulness": 0.91, "answer_relevancy": 0.82, "context_precision": 0.75,
         "latency_seconds": 2.1, "cost_usd": 0.0042, "error": None},
        {"faithfulness": 1.0,  "answer_relevancy": 0.88, "context_precision": 0.60,
         "latency_seconds": 1.2, "cost_usd": 0.0039, "error": None},
        {"faithfulness": 0.3,  "answer_relevancy": 0.79, "context_precision": 0.70,
         "latency_seconds": 3.8, "cost_usd": 0.0047, "error": None},   # hallucinated answer
    ]
    agg = compute_aggregate_metrics(mock_results)
    for k, v in agg.items():
        print(f"    {k:<22}: {v}")
    print()
    print_gate_report(check_gates(agg))

    print("Quality gate tests complete.\n")
