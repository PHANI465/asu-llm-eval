"""Offline tests for the quality gate engine — no API keys or network needed."""

import pytest

import quality_gates
from project_config import load_config, token_cost_usd

THRESHOLDS = {
    "hallucination_rate_max":  0.10,
    "answer_relevancy_min":    0.75,
    "faithfulness_min":        0.80,
    "context_precision_min":   0.60,
    "latency_p95_max_seconds": 15.0,
    "cost_per_query_max_usd":  0.02,
    "error_rate_max":          0.10,
}


def scored(faithfulness=0.9, relevancy=0.9, precision=0.9, latency=2.0, cost=0.004, error=None):
    return {
        "faithfulness": faithfulness, "answer_relevancy": relevancy, "context_precision": precision,
        "latency_seconds": latency, "cost_usd": cost, "error": error,
    }


def failed_call(error="RateLimitError: Error code: 429 - insufficient_quota"):
    return scored(faithfulness=None, relevancy=None, precision=None, latency=1.4, cost=None, error=error)


# ── hallucination rate ────────────────────────────────────────────────────────

def test_hallucination_rate_counts_answers_below_threshold():
    results = [scored(1.0), scored(0.9), scored(0.4), scored(0.0)]
    assert quality_gates.compute_hallucination_rate(results, threshold=0.5) == 0.5


def test_hallucination_rate_ignores_unscored_answers():
    results = [scored(1.0), scored(None), failed_call()]
    assert quality_gates.compute_hallucination_rate(results, threshold=0.5) == 0.0


def test_hallucination_rate_is_none_when_nothing_was_scored():
    assert quality_gates.compute_hallucination_rate([failed_call()] * 3, threshold=0.5) is None


def test_hallucination_rate_is_not_just_inverse_faithfulness():
    # Mean faithfulness 0.85 passes the faithfulness gate, but one answer in
    # four is fabricated — the hallucination gate must catch it independently.
    results = [scored(1.0), scored(1.0), scored(1.0), scored(0.4)]
    metrics = quality_gates.compute_aggregate_metrics(results, hallucination_faithfulness_threshold=0.5)
    gates = quality_gates.check_gates(metrics, THRESHOLDS)
    assert gates["gates"]["faithfulness"]["passed"]
    assert not gates["gates"]["hallucination_rate"]["passed"]


# ── aggregation ───────────────────────────────────────────────────────────────

def test_aggregate_metrics_happy_path():
    results = [scored(0.9, 0.8, 0.7, latency=1.0, cost=0.004), scored(1.0, 1.0, 0.9, latency=3.0, cost=0.006)]
    metrics = quality_gates.compute_aggregate_metrics(results, hallucination_faithfulness_threshold=0.5)
    assert metrics["faithfulness"] == 0.95
    assert metrics["answer_relevancy"] == 0.9
    assert metrics["context_precision"] == 0.8
    assert metrics["cost_per_query_usd"] == 0.005
    assert metrics["error_rate"] == 0.0
    assert metrics["hallucination_rate"] == 0.0
    assert 2.8 < metrics["latency_p95_seconds"] <= 3.0


def test_failed_calls_do_not_improve_latency_or_cost():
    results = [scored(latency=10.0, cost=0.01), failed_call(), failed_call()]
    metrics = quality_gates.compute_aggregate_metrics(results, hallucination_faithfulness_threshold=0.5)
    assert metrics["latency_p95_seconds"] == 10.0
    assert metrics["cost_per_query_usd"] == 0.01
    assert metrics["error_rate"] == round(2 / 3, 4)


def test_total_outage_fails_every_quality_gate():
    """Regression: an all-429 run used to report 0% hallucination and $0 cost, passing both gates."""
    metrics = quality_gates.compute_aggregate_metrics([failed_call()] * 10, hallucination_faithfulness_threshold=0.5)
    gates = quality_gates.check_gates(metrics, THRESHOLDS)

    assert gates["overall"] == "FAIL"
    assert set(gates["passed_gates"]) == set()
    assert metrics["error_rate"] == 1.0


def test_aggregate_rejects_empty_results():
    with pytest.raises(ValueError):
        quality_gates.compute_aggregate_metrics([])


# ── gate checks ───────────────────────────────────────────────────────────────

def test_gate_directions_and_boundaries():
    metrics = {
        "hallucination_rate": 0.10, "answer_relevancy": 0.75, "faithfulness": 0.79,
        "context_precision": 0.60, "latency_p95_seconds": 15.1, "cost_per_query_usd": 0.02,
        "error_rate": 0.0,
    }
    result = quality_gates.check_gates(metrics, THRESHOLDS)
    assert result["overall"] == "FAIL"
    assert result["failed_gates"] == ["faithfulness", "latency_p95"]
    assert result["gates"]["hallucination_rate"]["passed"]   # equal to max is allowed
    assert result["gates"]["answer_relevancy"]["passed"]     # equal to min is allowed


def test_missing_value_fails_and_missing_threshold_is_skipped():
    metrics = {key: 0.0 for key, *_ in quality_gates.GATE_DEFINITIONS}
    metrics["answer_relevancy"] = None
    thresholds = {k: v for k, v in THRESHOLDS.items() if k != "error_rate_max"}
    thresholds["answer_relevancy_min"] = 0.0
    thresholds["faithfulness_min"] = 0.0
    thresholds["context_precision_min"] = 0.0

    result = quality_gates.check_gates(metrics, thresholds)
    assert result["failed_gates"] == ["answer_relevancy"]
    assert result["gates"]["error_rate"]["message"].startswith("SKIP")
    assert result["gates"]["error_rate"]["passed"]


def test_every_gate_has_a_threshold_in_config():
    configured = load_config()["quality_gates"]
    for _, threshold_key, _, _ in quality_gates.GATE_DEFINITIONS:
        assert threshold_key in configured, f"config.yaml is missing quality_gates.{threshold_key}"


def test_models_in_config_have_prices():
    evaluation = load_config()["evaluation"]
    for model in (evaluation["model"], evaluation["judge_model"], evaluation["embedding_model"]):
        assert token_cost_usd(model, 1000, 1000) is not None, f"config.yaml pricing is missing {model}"


def test_token_cost_uses_separate_input_and_output_prices():
    prices = load_config()["pricing"]["gpt-4o"]
    expected = (1_000_000 * prices["input"] + 500_000 * prices["output"]) / 1_000_000
    assert token_cost_usd("gpt-4o", 1_000_000, 500_000) == pytest.approx(expected)
    assert token_cost_usd("some-unpriced-model", 10, 10) is None
