"""Offline tests for question selection, report building and result persistence."""

import json
import os
import sqlite3
from collections import Counter

import pytest

import run_eval
from project_config import KNOWLEDGE_BASE_DIR


@pytest.fixture(scope="module")
def dataset():
    return run_eval.load_golden_dataset()


def record(id_, category, faithfulness=0.95, error=None, evaluation_error=None, cost=0.004):
    return {
        "id": id_, "question": f"Question {id_}?", "answer": "" if error else f"Answer {id_}",
        "expected_answer": f"Expected {id_}", "category": category, "difficulty": "easy",
        "faithfulness": None if error else faithfulness,
        "answer_relevancy": None if error else 0.9,
        "context_precision": None if error else 0.9,
        "latency_seconds": 1.0 if error else 3.0,
        "token_usage": 0 if error else 1500,
        "cost_usd": None if error else cost,
        "error": error, "evaluation_error": evaluation_error,
    }


RUN_META = {
    "run_timestamp": "2026-09-12T10:00:00+00:00", "commit_id": "abc1234", "full_run": False,
    "dataset_size": 100, "models": {"answer": "gpt-4o", "judge": "gpt-4o-mini", "embedding": "e"},
}


# ── golden dataset ────────────────────────────────────────────────────────────

def test_golden_dataset_is_well_formed(dataset):
    ids = [item["id"] for item in dataset]
    assert len(ids) == len(set(ids)), "duplicate ids in golden_dataset.json"
    kb_files = set(os.listdir(KNOWLEDGE_BASE_DIR))
    for item in dataset:
        for field in ("question", "expected_answer", "category", "difficulty", "source_document"):
            assert item.get(field), f"question {item['id']} is missing '{field}'"
        assert item["source_document"] in kb_files, f"question {item['id']} cites a missing document"


# ── question selection ────────────────────────────────────────────────────────

def test_test_mode_sample_covers_every_category(dataset):
    picked = run_eval.select_questions(dataset, 10)
    all_categories = {item["category"] for item in dataset}

    assert len(picked) == 10
    assert {item["category"] for item in picked} == all_categories
    assert max(Counter(item["category"] for item in picked).values()) <= 2


def test_selection_is_deterministic_and_keeps_dataset_order(dataset):
    first = run_eval.select_questions(dataset, 10)
    assert first == run_eval.select_questions(dataset, 10)
    ids = [item["id"] for item in first]
    assert ids == sorted(ids)


def test_full_selection_returns_everything(dataset):
    assert run_eval.select_questions(dataset, None) == dataset
    assert run_eval.select_questions(dataset, len(dataset) + 5) == dataset


def test_selection_rejects_non_positive_limit(dataset):
    with pytest.raises(ValueError):
        run_eval.select_questions(dataset, 0)


def test_full_flag_from_environment(monkeypatch):
    monkeypatch.setenv("EVAL_FULL_RUN", "true")
    assert run_eval.parse_args([]).full
    monkeypatch.setenv("EVAL_FULL_RUN", "false")
    assert not run_eval.parse_args([]).full
    assert run_eval.parse_args(["--full"]).full


# ── report building ───────────────────────────────────────────────────────────

def build(records):
    metrics = run_eval.quality_gates.compute_aggregate_metrics(records)
    gates = run_eval.quality_gates.check_gates(metrics)
    usage = {"input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.0003}
    return run_eval.build_report(RUN_META, records, metrics, gates, usage)


def test_report_keeps_category_and_flags_failures():
    records = [
        record(1, "tuition"),
        record(2, "housing", faithfulness=0.2),
        record(3, "admissions", error="RateLimitError: 429"),
        record(4, "housing", faithfulness=None, evaluation_error="RAGAS returned no scores"),
    ]
    report = build(records)

    assert [r["passed"] for r in report["all_results"]] == [True, False, False, False]
    assert report["failed_question_count"] == 3

    failures = report["sample_failures"]
    assert [f["id"] for f in failures] == [3, 4, 2]          # errors first, then worst faithfulness
    assert failures[0]["error"] == "RateLimitError: 429"
    assert failures[0]["category"] == "admissions"           # regression: category used to be blank
    assert failures[2]["expected_answer"] == "Expected 2"


def test_report_costs_and_metadata():
    report = build([record(1, "tuition", cost=0.004), record(2, "housing", cost=0.006)])

    assert report["answer_cost_usd"] == 0.01
    assert report["judge_cost_usd"] == 0.0003
    assert report["total_cost_usd"] == 0.0103
    assert report["total_tokens"] == 3000 + 1200
    assert report["dataset_size"] == 100
    assert report["test_mode"] is True
    assert report["overall_result"] in ("PASS", "FAIL")
    assert report["models"]["judge"] == "gpt-4o-mini"


def test_error_report_has_hint_and_empty_gates():
    report = run_eval.build_error_report(
        RUN_META, "RateLimitError: Error code: 429 - {'code': 'insufficient_quota'}", "tb"
    )
    assert report["overall_result"] == "ERROR"
    assert report["gate_results"]["gates"] == {}
    assert "credit" in report["error_hint"]


@pytest.mark.parametrize("error, fragment", [
    ("AuthenticationError: Error code: 401 - invalid_api_key", "rejected"),
    ("OSError: OPENAI_API_KEY not found. Set it as an environment variable", "missing"),
    ("ValueError: something unrelated", ""),
])
def test_error_hints(error, fragment):
    hint = run_eval.error_hint(error)
    assert (fragment in hint) if fragment else hint == ""


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_all_writes_report_history_and_database(tmp_path):
    report = build([record(1, "tuition"), record(2, "housing")])
    run_eval.save_all(report, str(tmp_path))
    run_eval.save_all(run_eval.build_error_report(RUN_META, "PipelineError: boom", ""), str(tmp_path))

    latest = json.loads((tmp_path / "latest_report.json").read_text(encoding="utf-8"))
    assert latest["overall_result"] == "ERROR"

    runs = json.loads((tmp_path / "eval_history.json").read_text(encoding="utf-8"))["runs"]
    assert [r["overall_result"] for r in runs] == [report["overall_result"], "ERROR"]
    assert "error_rate" in runs[0]

    with sqlite3.connect(tmp_path / "eval_history.db") as conn:
        rows = conn.execute("SELECT overall_result, error_rate, pipeline_error FROM eval_runs ORDER BY id").fetchall()
    assert rows[1] == ("ERROR", None, "PipelineError: boom")


def test_database_migrates_old_schema(tmp_path):
    with sqlite3.connect(tmp_path / "eval_history.db") as conn:
        conn.execute("CREATE TABLE eval_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_timestamp TEXT, overall_result TEXT)")
        conn.execute("INSERT INTO eval_runs (run_timestamp, overall_result) VALUES ('old', 'PASS')")

    run_eval.save_to_database(run_eval.history_row(build([record(1, "tuition")])), str(tmp_path))

    with sqlite3.connect(tmp_path / "eval_history.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM eval_runs WHERE error_rate IS NOT NULL").fetchone()[0] == 1


def test_history_is_capped(tmp_path):
    row = run_eval.history_row(build([record(1, "tuition")]))
    for _ in range(run_eval.MAX_HISTORY_RUNS + 3):
        run_eval.save_history_json(row, str(tmp_path))
    runs = json.loads((tmp_path / "eval_history.json").read_text(encoding="utf-8"))["runs"]
    assert len(runs) == run_eval.MAX_HISTORY_RUNS
