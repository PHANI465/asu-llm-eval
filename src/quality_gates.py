# =============================================================================
# src/quality_gates.py
# ASU LLM Evaluation — Quality Gate Engine
#
# Responsibilities:
#   1. compute_hallucination_rate(results) -> float
#         Derived from faithfulness: hallucination = 1 - avg_faithfulness
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

import os
import sys

import numpy as np
import yaml

# -----------------------------------------------------------------------------
# 0. Load project config (thresholds live in config.yaml → quality_gates)
# -----------------------------------------------------------------------------

# Walk up one directory from src/ to reach the project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIG_PATH  = os.path.join(_PROJECT_ROOT, "config.yaml")

def _load_thresholds() -> dict:
    """
    Load quality-gate thresholds from config.yaml.
    Returns the quality_gates section as a plain dict.
    """
    try:
        with open(_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg["quality_gates"]
    except FileNotFoundError:
        raise FileNotFoundError(f"config.yaml not found at: {_CONFIG_PATH}")
    except KeyError:
        raise KeyError("config.yaml is missing the 'quality_gates' section.")


# -----------------------------------------------------------------------------
# 1. Hallucination rate (derived metric)
# -----------------------------------------------------------------------------

def compute_hallucination_rate(results: list) -> float:
    """
    Compute the hallucination rate across all evaluated results.

    Definition
    ----------
    hallucination_rate = 1.0 - average(faithfulness)

    A faithfulness score of 1.0 means the answer came entirely from the
    retrieved context (no hallucination).  A score of 0.0 means the answer
    was entirely fabricated.  Inverting gives us a hallucination rate that
    is intuitive: 0 = no hallucination, 1 = total hallucination.

    Parameters
    ----------
    results : list of scored dicts from evaluator.evaluate_batch()

    Returns
    -------
    float in [0, 1], or 0.0 if no valid faithfulness scores exist
    """
    scores = [
        r["faithfulness"]
        for r in results
        if r.get("faithfulness") is not None
    ]

    if not scores:
        return 0.0  # cannot determine hallucination — default to 0 (optimistic)

    avg_faithfulness = sum(scores) / len(scores)
    return round(max(0.0, min(1.0, 1.0 - avg_faithfulness)), 4)


# -----------------------------------------------------------------------------
# 2. Aggregate metrics across the full result set
# -----------------------------------------------------------------------------

# Approximate GPT-4o blended cost per token
# (Input: ~$5/M tokens, Output: ~$15/M tokens → blended ~$10/M = $0.00001/token)
# The user-specified simpler approximation is $0.000005/token — used here.
_COST_PER_TOKEN_USD = 0.000005


def compute_aggregate_metrics(results: list) -> dict:
    """
    Compute aggregate metrics from a list of scored result dicts.

    Parameters
    ----------
    results : list of dicts from evaluator.evaluate_batch()
              Each dict must contain:
                faithfulness, answer_relevancy, context_precision,
                latency_seconds, token_usage
              None values are skipped per metric.

    Returns
    -------
    dict with keys:
        hallucination_rate    float  — derived from faithfulness
        answer_relevancy      float  — mean answer relevancy
        faithfulness          float  — mean faithfulness
        context_precision     float  — mean context precision
        latency_p95_seconds   float  — 95th-percentile latency (numpy)
        cost_per_query_usd    float  — mean cost estimate per query
    """
    if not results:
        raise ValueError("results list is empty — nothing to aggregate.")

    # --- Helper: collect non-None values for a key ---
    def _collect(key: str) -> list:
        return [r[key] for r in results if r.get(key) is not None]

    def _mean(values: list) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    # --- Individual metric averages ---
    faithfulness_scores     = _collect("faithfulness")
    relevancy_scores        = _collect("answer_relevancy")
    precision_scores        = _collect("context_precision")

    avg_faithfulness    = _mean(faithfulness_scores)
    avg_relevancy       = _mean(relevancy_scores)
    avg_precision       = _mean(precision_scores)

    # --- Hallucination rate (derived from faithfulness) ---
    hallucination_rate = compute_hallucination_rate(results)

    # --- Latency p95 using numpy ---
    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds") is not None]
    if latencies:
        latency_p95 = round(float(np.percentile(latencies, 95)), 3)
    else:
        latency_p95 = None

    # --- Cost per query ---
    # Estimate: token_usage * cost_per_token
    costs = [
        r["token_usage"] * _COST_PER_TOKEN_USD
        for r in results
        if r.get("token_usage") is not None
    ]
    avg_cost = round(sum(costs) / len(costs), 6) if costs else None

    return {
        "hallucination_rate":  hallucination_rate,
        "answer_relevancy":    avg_relevancy,
        "faithfulness":        avg_faithfulness,
        "context_precision":   avg_precision,
        "latency_p95_seconds": latency_p95,
        "cost_per_query_usd":  avg_cost,
    }


# -----------------------------------------------------------------------------
# 3. Gate evaluation
# -----------------------------------------------------------------------------

def check_gates(metrics: dict) -> dict:
    """
    Evaluate each aggregate metric against its configured threshold.

    Parameters
    ----------
    metrics : dict from compute_aggregate_metrics() with keys:
                hallucination_rate, answer_relevancy, faithfulness,
                context_precision, latency_p95_seconds, cost_per_query_usd

    Returns
    -------
    dict:
        overall       : "PASS" or "FAIL"
        gates         : dict of per-gate result dicts
        failed_gates  : list of gate names that failed
        passed_gates  : list of gate names that passed
    """
    thresholds = _load_thresholds()

    # ------------------------------------------------------------------
    # Gate definitions:
    #   Each entry is (metric_key, threshold_key, direction, label)
    #   direction: "max" = value must be BELOW threshold (lower is better)
    #              "min" = value must be ABOVE threshold (higher is better)
    # ------------------------------------------------------------------
    gate_definitions = [
        ("hallucination_rate",  "hallucination_rate_max",    "max", "hallucination_rate"),
        ("answer_relevancy",    "answer_relevancy_min",       "min", "answer_relevancy"),
        ("faithfulness",        "faithfulness_min",           "min", "faithfulness"),
        ("context_precision",   "context_precision_min",      "min", "context_precision"),
        ("latency_p95_seconds", "latency_p95_max_seconds",   "max", "latency_p95"),
        ("cost_per_query_usd",  "cost_per_query_max_usd",    "max", "cost_per_query"),
    ]

    gates        = {}
    passed_gates = []
    failed_gates = []

    for metric_key, threshold_key, direction, gate_name in gate_definitions:
        value     = metrics.get(metric_key)
        threshold = thresholds.get(threshold_key)

        # Handle missing or None values gracefully
        if value is None:
            gates[gate_name] = {
                "value":     None,
                "threshold": threshold,
                "passed":    False,
                "message":   f"SKIP: no value available for '{metric_key}'",
            }
            failed_gates.append(gate_name)
            continue

        if threshold is None:
            gates[gate_name] = {
                "value":     value,
                "threshold": None,
                "passed":    True,   # no threshold configured → do not block
                "message":   f"SKIP: no threshold configured for '{threshold_key}'",
            }
            passed_gates.append(gate_name)
            continue

        # Evaluate the gate
        if direction == "max":
            passed = value <= threshold
            # Message describes the actual relationship, not the expectation
            op_word = "below max" if passed else "exceeds max"
            op_sym  = "<="
        else:  # direction == "min"
            passed = value >= threshold
            op_word = "above min" if passed else "below min"
            op_sym  = ">="

        verdict = "PASS" if passed else "FAIL"
        message = f"{verdict}: {value} {op_word} {threshold}"

        gates[gate_name] = {
            "value":     value,
            "threshold": threshold,
            "passed":    passed,
            "message":   message,
        }

        (passed_gates if passed else failed_gates).append(gate_name)

    overall = "PASS" if len(failed_gates) == 0 else "FAIL"

    return {
        "overall":      overall,
        "gates":        gates,
        "failed_gates": failed_gates,
        "passed_gates": passed_gates,
    }


# -----------------------------------------------------------------------------
# 4. Human-readable report printer
# -----------------------------------------------------------------------------

def print_gate_report(gate_results: dict) -> None:
    """
    Print a formatted quality-gate report to stdout.

    Parameters
    ----------
    gate_results : dict returned by check_gates()
    """
    DIVIDER = "=" * 44

    # ------------------------------------------------------------------
    # Display configuration:
    #   (gate_name, format_fn for value, format_fn for threshold, symbol)
    # ------------------------------------------------------------------
    _fmt = {
        "hallucination_rate": (lambda v: f"{v:.4f}",  lambda t: f"{t}",    "<="),
        "answer_relevancy":   (lambda v: f"{v:.4f}",  lambda t: f"{t}",    ">="),
        "faithfulness":       (lambda v: f"{v:.4f}",  lambda t: f"{t}",    ">="),
        "context_precision":  (lambda v: f"{v:.4f}",  lambda t: f"{t}",    ">="),
        "latency_p95":        (lambda v: f"{v}s",     lambda t: f"{t}s",   "<="),
        "cost_per_query":     (lambda v: f"${v:.4f}", lambda t: f"${t}",   "<="),
    }

    gates = gate_results["gates"]

    print(DIVIDER)
    print("QUALITY GATE REPORT")
    print(DIVIDER)

    for gate_name, info in gates.items():
        verdict   = "PASS" if info["passed"] else "FAIL"
        value     = info["value"]
        threshold = info["threshold"]

        if value is None or threshold is None:
            detail = info["message"]
        else:
            fmt_v, fmt_t, sym = _fmt.get(
                gate_name,
                (lambda v: str(v), lambda t: str(t), "?")
            )
            detail = f"({fmt_v(value)} {sym} {fmt_t(threshold)})"

        # Pad gate name for alignment
        label = f"{gate_name:<20}"
        print(f"  {label}: {verdict:<4}  {detail}")

    print(DIVIDER)

    overall      = gate_results["overall"]
    failed_gates = gate_results["failed_gates"]
    passed_gates = gate_results["passed_gates"]

    print(f"OVERALL RESULT: {overall}")

    if overall == "PASS":
        print(f"All {len(passed_gates)} gates passed. Safe to deploy.")
    else:
        n_failed = len(failed_gates)
        print(f"{n_failed} gate(s) failed. Deployment blocked.")
        print(f"Failed gates: {', '.join(failed_gates)}")

    print(DIVIDER)


# -----------------------------------------------------------------------------
# 5. Quick-test entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 44)
    print("  Quality Gates - Scenario Tests")
    print("=" * 44)

    # ------------------------------------------------------------------
    # Scenario 1: All gates pass
    # ------------------------------------------------------------------
    print("\n--- Scenario 1: All gates should PASS ---\n")

    metrics_pass = {
        "hallucination_rate":  0.03,
        "answer_relevancy":    0.82,
        "faithfulness":        0.91,
        "context_precision":   0.75,
        "latency_p95_seconds": 2.1,
        "cost_per_query_usd":  0.014,
    }

    gate_results_pass = check_gates(metrics_pass)
    print_gate_report(gate_results_pass)

    # ------------------------------------------------------------------
    # Scenario 2: Two gates fail (hallucination_rate + answer_relevancy)
    # ------------------------------------------------------------------
    print("\n--- Scenario 2: Two gates should FAIL ---\n")

    metrics_fail = {
        "hallucination_rate":  0.12,   # FAIL: 0.12 > max 0.05
        "answer_relevancy":    0.60,   # FAIL: 0.60 < min 0.75
        "faithfulness":        0.91,
        "context_precision":   0.75,
        "latency_p95_seconds": 2.1,
        "cost_per_query_usd":  0.014,
    }

    gate_results_fail = check_gates(metrics_fail)
    print_gate_report(gate_results_fail)

    # ------------------------------------------------------------------
    # Also demonstrate compute_aggregate_metrics with mock data
    # ------------------------------------------------------------------
    print("\n--- Bonus: compute_aggregate_metrics demo ---\n")

    mock_results = [
        {
            "question":          "Q1",
            "answer":            "A1",
            "faithfulness":      0.91,
            "answer_relevancy":  0.82,
            "context_precision": 0.75,
            "latency_seconds":   2.1,
            "token_usage":       400,
            "evaluation_error":  None,
        },
        {
            "question":          "Q2",
            "answer":            "A2",
            "faithfulness":      1.0,
            "answer_relevancy":  0.88,
            "context_precision": 0.60,
            "latency_seconds":   1.2,
            "token_usage":       350,
            "evaluation_error":  None,
        },
        {
            "question":          "Q3",
            "answer":            "A3",
            "faithfulness":      0.85,
            "answer_relevancy":  0.79,
            "context_precision": 0.70,
            "latency_seconds":   3.8,   # this one is slow — will push p95 up
            "token_usage":       500,
            "evaluation_error":  None,
        },
    ]

    agg = compute_aggregate_metrics(mock_results)
    print("  Aggregate metrics from 3 mock results:")
    for k, v in agg.items():
        print(f"    {k:<22}: {v}")

    print("\n  Running check_gates on aggregated metrics...")
    gate_results_agg = check_gates(agg)
    print_gate_report(gate_results_agg)

    print("Quality gate tests complete.\n")
