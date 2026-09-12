# =============================================================================
# src/run_eval.py
# ASU LLM Evaluation — Master Evaluation Controller
#
# This is the file GitHub Actions runs on every push.
# It wires together:
#   rag_pipeline  → answers the selected golden-dataset questions
#   evaluator     → scores each answer with RAGAS
#   quality_gates → aggregates metrics + runs PASS/FAIL gates
#   SQLite        → archives every run to results/eval_history.db (local only)
#   JSON          → writes results/latest_report.json + eval_history.json
#                   (committed by CI and read by both dashboards)
#
# Usage:
#   python src/run_eval.py                 # test run: 10 questions across all categories
#   python src/run_eval.py --full          # every question (or set EVAL_FULL_RUN=true)
#   python src/run_eval.py --limit 3       # quick, cheap smoke test
#   python src/run_eval.py --results-dir /tmp/eval   # keep experiments out of results/
#
# Result statuses / exit codes (critical for GitHub Actions):
#   PASS  → exit 0  all quality gates met
#   FAIL  → exit 1  one or more quality gates missed
#   ERROR → exit 1  the pipeline could not run (bad key, exhausted quota, crash);
#                   says nothing about answer quality
# =============================================================================

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# rag_pipeline / evaluator pull in LangChain + RAGAS, so they are imported
# inside the functions that use them: any import-time failure then lands in
# the crash handler, which still writes an ERROR report for the dashboard.
import quality_gates
from project_config import GOLDEN_DATASET_PATH, PROJECT_ROOT, RESULTS_DIR, load_config

MAX_HISTORY_RUNS      = 50
SAMPLE_FAILURE_LIMIT  = 5
PREFLIGHT_QUESTION    = "What campuses does ASU have?"

METRIC_KEYS = [metric_key for metric_key, *_ in quality_gates.GATE_DEFINITIONS]


class PipelineError(RuntimeError):
    """The evaluation could not run (keys, quota, network) — reported as ERROR, not FAIL."""


# =============================================================================
# Utility helpers
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASU RAG evaluation and quality gates.")
    parser.add_argument(
        "--full", action="store_true",
        default=os.getenv("EVAL_FULL_RUN", "").strip().lower() in ("1", "true", "yes"),
        help="Evaluate every golden-dataset question (also enabled by EVAL_FULL_RUN=true).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Questions in a non-full run (default: evaluation.test_mode_questions in config.yaml).",
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR,
        help="Directory for reports (default: results/). Use another path for local experiments.",
    )
    return parser.parse_args(argv)


def _format_duration(seconds: float) -> str:
    """Convert a raw second count to a readable string like '4m 32s'."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s" if m > 0 else f"{s}s"


def _get_commit_id() -> str:
    """
    Return the short git commit hash of HEAD.
    Falls back to GITHUB_SHA, then 'local' if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return os.getenv("GITHUB_SHA", "")[:7] or "local"


def _safe_pct(val) -> str:
    """Format a float as a percentage string, handling None gracefully."""
    try:
        return f"{float(val):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def _safe_val(val) -> str:
    """Format a numeric value for display, handling None gracefully."""
    if val is None:
        return "N/A"
    try:
        return str(round(float(val), 4))
    except (TypeError, ValueError):
        return "N/A"


def error_hint(error: str) -> str:
    """Translate common infrastructure failures into an actionable hint."""
    text = (error or "").lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return ("The OpenAI account behind OPENAI_API_KEY has run out of credit. Add credits/billing "
                "at platform.openai.com (in CI this is the GitHub secret's key).")
    if "invalid_api_key" in text or "incorrect api key" in text or "unauthorized" in text or "error code: 401" in text:
        return "An API key was rejected. Check OPENAI_API_KEY and PINECONE_API_KEY."
    if "not found. set it" in text:
        return "An API key is missing. Set it in .env locally, or as a repository secret in CI."
    if "rate limit" in text or "error code: 429" in text:
        return "Rate limited by OpenAI. Lower evaluation.judge_max_workers in config.yaml or retry later."
    return ""


# =============================================================================
# Step 1 — Load golden dataset + choose questions
# =============================================================================

def load_golden_dataset() -> list:
    """Load questions + expected answers from data/golden_dataset.json."""
    if not os.path.exists(GOLDEN_DATASET_PATH):
        raise FileNotFoundError(
            f"Golden dataset not found at: {GOLDEN_DATASET_PATH}\n"
            "Make sure data/golden_dataset.json is populated."
        )

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("golden_dataset.json is empty.")
    return dataset


def select_questions(dataset: list, limit: int | None) -> list:
    """
    Pick `limit` questions spread evenly across categories (round-robin in
    dataset order), so a quick run covers every topic instead of only the
    first category. Deterministic, so runs stay comparable over time.
    Returns the whole dataset when limit is None or covers everything.
    """
    if limit is None or limit >= len(dataset):
        return list(dataset)
    if limit < 1:
        raise ValueError("--limit must be at least 1")

    queues = {}
    for item in dataset:
        queues.setdefault(item.get("category", ""), []).append(item)

    picked = []
    while len(picked) < limit:
        for queue in queues.values():
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))

    order = {id(item): pos for pos, item in enumerate(dataset)}
    return sorted(picked, key=lambda item: order[id(item)])


# =============================================================================
# Step 2 — RAG pipeline (answer every question)
# =============================================================================

def run_preflight() -> None:
    """
    Ask one throwaway question before the timed run. It absorbs the cold start
    (documents, chunking, Pinecone sync) so latency numbers are fair, and it
    stops the run before spending money when keys or quota are broken.
    """
    import rag_pipeline

    print("\n[Preflight] Checking API access with one warm-up question (not counted in results)...")
    warmup = rag_pipeline.get_answer(PREFLIGHT_QUESTION)
    if warmup["error"]:
        raise PipelineError(f"Preflight question failed, evaluation not started. {warmup['error']}")
    print(f"[Preflight] OK ({warmup['latency_seconds']}s). Pipeline is warm.")


def run_rag_pipeline(questions: list) -> list:
    """
    Pass every question through rag_pipeline.get_answer() sequentially (so
    latency reflects a single user). Returns one record per question that
    combines golden-dataset fields with the RAG result.
    """
    import rag_pipeline

    total   = len(questions)
    records = []

    print(f"\n[Step 2/5] Running RAG pipeline ({total} question(s))...")

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        result   = rag_pipeline.get_answer(question)

        status = f"ERROR {result['error'][:120]}" if result["error"] else f"ok {result['latency_seconds']}s"
        print(f"  [{i}/{total}] {question[:70]}  -> {status}")

        records.append({
            "id":              item.get("id"),
            "category":        item.get("category", ""),
            "difficulty":      item.get("difficulty", ""),
            "expected_answer": item.get("expected_answer", ""),
            **result,
        })

    errors = [r for r in records if r["error"]]
    if errors and len(errors) == total:
        raise PipelineError(f"All {total} questions failed in the RAG step. First error: {errors[0]['error']}")
    if errors:
        print(f"  [WARNING] {len(errors)}/{total} question(s) failed in the RAG step (see error_rate gate).")

    return records


# =============================================================================
# Step 3 — RAGAS evaluation (score every answer)
# =============================================================================

def run_evaluation(records: list) -> dict:
    """
    Score every record with evaluator.evaluate_batch(), using each question's
    expected_answer as the reference so ContextPrecision (with reference) is
    used. Scores are merged into the records in place.
    Returns the judge's token usage / cost.
    """
    import evaluator

    print(f"\n[Step 3/5] Scoring {len(records)} answer(s) with RAGAS...")

    scores, judge_usage = evaluator.evaluate_batch(
        records, references=[r["expected_answer"] for r in records]
    )
    for record, score in zip(records, scores):
        record.update({
            "faithfulness":      score["faithfulness"],
            "answer_relevancy":  score["answer_relevancy"],
            "context_precision": score["context_precision"],
            "evaluation_error":  score["evaluation_error"],
        })
    return judge_usage


# =============================================================================
# Steps 4 + 5 — Aggregate metrics, quality gates
# =============================================================================

def build_metrics(records: list) -> dict:
    """Delegate to quality_gates.compute_aggregate_metrics() and log the results."""
    print("\n[Step 4/5] Computing aggregate metrics...")
    metrics = quality_gates.compute_aggregate_metrics(records)

    print("  Aggregated metrics:")
    for key, val in metrics.items():
        print(f"    {key:<24}: {_safe_val(val)}")

    return metrics


def run_quality_gates(metrics: dict) -> dict:
    """Run quality gates and print the gate report."""
    print("\n[Step 5/5] Running quality gates...")
    gate_results = quality_gates.check_gates(metrics)
    quality_gates.print_gate_report(gate_results)
    return gate_results


# =============================================================================
# Report building
# =============================================================================

def question_passed(record: dict, threshold: float) -> bool:
    """A question passes when its RAG call succeeded and its answer is faithful enough."""
    faith = record.get("faithfulness")
    return not record.get("error") and faith is not None and faith >= threshold


def build_report(run_meta: dict, records: list, metrics: dict, gate_results: dict, judge_usage: dict) -> dict:
    """Assemble the full JSON report consumed by CI and both dashboards."""
    threshold = quality_gates.hallucination_threshold()

    all_results = []
    for r in records:
        all_results.append({
            "id":                r.get("id"),
            "question":          r.get("question", ""),
            "answer":            (r.get("answer") or "")[:500],
            "category":          r.get("category", ""),
            "difficulty":        r.get("difficulty", ""),
            "faithfulness":      r.get("faithfulness"),
            "answer_relevancy":  r.get("answer_relevancy"),
            "context_precision": r.get("context_precision"),
            "latency_seconds":   r.get("latency_seconds"),
            "token_usage":       r.get("token_usage"),
            "cost_usd":          r.get("cost_usd"),
            "error":             r.get("error") or r.get("evaluation_error"),
            "passed":            question_passed(r, threshold),
        })

    # Failed questions, worst first: RAG errors, then unscored, then lowest faithfulness
    failed = [r for r in records if not question_passed(r, threshold)]
    failed.sort(key=lambda r: (
        0 if r.get("error") else 1,
        r["faithfulness"] if r.get("faithfulness") is not None else -1.0,
    ))
    sample_failures = [
        {
            "id":               r.get("id"),
            "question":         r.get("question", ""),
            "expected_answer":  r.get("expected_answer", ""),
            "answer":           (r.get("answer") or "")[:500],
            "faithfulness":     r.get("faithfulness"),
            "answer_relevancy": r.get("answer_relevancy"),
            "category":         r.get("category", ""),
            "difficulty":       r.get("difficulty", ""),
            "error":            r.get("error") or r.get("evaluation_error"),
        }
        for r in failed[:SAMPLE_FAILURE_LIMIT]
    ]

    answer_tokens = sum(r.get("token_usage") or 0 for r in records)
    answer_cost   = sum(r.get("cost_usd") or 0 for r in records)
    judge_tokens  = (judge_usage.get("input_tokens") or 0) + (judge_usage.get("output_tokens") or 0)
    judge_cost    = judge_usage.get("cost_usd") or 0

    return {
        "run_timestamp":         run_meta["run_timestamp"],
        "commit_id":             run_meta["commit_id"],
        "total_questions":       len(records),
        "dataset_size":          run_meta.get("dataset_size"),
        "test_mode":             not run_meta["full_run"],
        "models":                run_meta.get("models", {}),
        "hallucination_faithfulness_threshold": threshold,
        "total_cost_usd":        round(answer_cost + judge_cost, 6),
        "answer_cost_usd":       round(answer_cost, 6),
        "judge_cost_usd":        round(judge_cost, 6),
        "total_tokens":          answer_tokens + judge_tokens,
        "metrics":               metrics,
        "gate_results": {
            "overall":      gate_results["overall"],
            "passed_gates": gate_results["passed_gates"],
            "failed_gates": gate_results["failed_gates"],
            "gates":        gate_results["gates"],
        },
        "overall_result":        gate_results["overall"],
        "failed_gates":          gate_results["failed_gates"],
        "failed_question_count": len(failed),
        "sample_failures":       sample_failures,
        "all_results":           all_results,
    }


def build_error_report(run_meta: dict, error: str, tb: str) -> dict:
    """Minimal report for a run that could not complete, so CI and dashboards still show why."""
    return {
        "run_timestamp":         run_meta["run_timestamp"],
        "commit_id":             run_meta["commit_id"],
        "total_questions":       0,
        "dataset_size":          run_meta.get("dataset_size"),
        "test_mode":             not run_meta["full_run"],
        "models":                run_meta.get("models", {}),
        "total_cost_usd":        0.0,
        "total_tokens":          0,
        "overall_result":        "ERROR",
        "failed_gates":          [],
        "metrics":               {},
        "gate_results": {
            "overall":      "ERROR",
            "passed_gates": [],
            "failed_gates": [],
            "gates":        {},
        },
        "failed_question_count": 0,
        "sample_failures":       [],
        "all_results":           [],
        "pipeline_error":        error,
        "error_hint":            error_hint(error),
        "traceback":             tb,
    }


def history_row(report: dict) -> dict:
    """Flat per-run summary shared by eval_history.json and the SQLite archive."""
    metrics = report.get("metrics") or {}
    return {
        "run_timestamp":   report["run_timestamp"],
        "commit_id":       report["commit_id"],
        "total_questions": report["total_questions"],
        "test_mode":       report["test_mode"],
        **{key: metrics.get(key) for key in METRIC_KEYS},
        "total_cost_usd":  report.get("total_cost_usd"),
        "total_tokens":    report.get("total_tokens"),
        "overall_result":  report["overall_result"],
        "failed_gates":    report.get("failed_gates", []),
        "pipeline_error":  report.get("pipeline_error"),
    }


# =============================================================================
# Save results
# =============================================================================

_DB_COLUMNS = [
    ("run_timestamp",       "TEXT"),
    ("commit_id",           "TEXT"),
    ("total_questions",     "INTEGER"),
    ("hallucination_rate",  "REAL"),
    ("answer_relevancy",    "REAL"),
    ("faithfulness",        "REAL"),
    ("context_precision",   "REAL"),
    ("latency_p95_seconds", "REAL"),
    ("cost_per_query_usd",  "REAL"),
    ("overall_result",      "TEXT"),
    ("failed_gates",        "TEXT"),
    ("error_rate",          "REAL"),
    ("total_cost_usd",      "REAL"),
    ("test_mode",           "INTEGER"),
    ("pipeline_error",      "TEXT"),
]


def save_to_database(row: dict, results_dir: str) -> None:
    """
    Append one run to eval_history.db (a local, gitignored archive).
    Creates the table on first run and adds columns introduced since.
    """
    os.makedirs(results_dir, exist_ok=True)

    conn = sqlite3.connect(os.path.join(results_dir, "eval_history.db"))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS eval_runs (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        existing = {info[1] for info in conn.execute("PRAGMA table_info(eval_runs)")}
        for name, sql_type in _DB_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE eval_runs ADD COLUMN {name} {sql_type}")

        values = {
            **{name: row.get(name) for name, _ in _DB_COLUMNS},
            "failed_gates": ", ".join(row.get("failed_gates") or []),
            "test_mode":    int(bool(row.get("test_mode"))),
        }
        names  = [name for name, _ in _DB_COLUMNS]
        cursor = conn.execute(
            f"INSERT INTO eval_runs ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
            [values[name] for name in names],
        )
        conn.commit()
        print(f"  [OK] eval_history.db — row inserted (id={cursor.lastrowid})")
    finally:
        conn.close()


def save_json_report(report: dict, results_dir: str) -> None:
    """Write the full report to latest_report.json."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "latest_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [OK] latest_report.json — {len(report['all_results'])} result(s) -> {path}")


def save_history_json(row: dict, results_dir: str) -> None:
    """Append a run summary to eval_history.json, keeping the most recent runs."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "eval_history.json")

    history = {"runs": []}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded.get("runs"), list):
                history = loaded
        except Exception as exc:
            print(f"  [WARNING] Could not read eval_history.json: {exc} — starting fresh.")

    history["runs"] = (history["runs"] + [row])[-MAX_HISTORY_RUNS:]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  [OK] eval_history.json — {len(history['runs'])} run(s) total -> {path}")


def save_all(report: dict, results_dir: str) -> None:
    """Write every output; one failing writer does not prevent the others."""
    print("\n[Saving results...]")
    row = history_row(report)
    for writer, target in (
        (save_json_report,  report),
        (save_history_json, row),
        (save_to_database,  row),
    ):
        try:
            writer(target, results_dir)
        except Exception as exc:
            print(f"  [WARNING] {writer.__name__} failed: {exc}")


# =============================================================================
# Final summary
# =============================================================================

def print_final_summary(report: dict, elapsed: float) -> None:
    """Print the human-readable evaluation summary to stdout."""
    DIVIDER = "=" * 44
    metrics = report["metrics"]

    print(f"\n{DIVIDER}")
    print("EVALUATION COMPLETE")
    print(DIVIDER)
    print(f"  Questions       : {report['total_questions']} of {report['dataset_size']}"
          + (" (test mode)" if report["test_mode"] else ""))
    print(f"  Time taken      : {_format_duration(elapsed)}")
    print(f"  Commit          : {report['commit_id']}")
    print(DIVIDER)
    print(f"  Hallucination   : {_safe_pct(metrics.get('hallucination_rate'))}")
    print(f"  Answer Relevancy: {_safe_val(metrics.get('answer_relevancy'))}")
    print(f"  Faithfulness    : {_safe_val(metrics.get('faithfulness'))}")
    print(f"  Context Prec    : {_safe_val(metrics.get('context_precision'))}")
    print(f"  Latency p95     : {_safe_val(metrics.get('latency_p95_seconds'))}s")
    print(f"  Cost per query  : ${metrics.get('cost_per_query_usd') or 0:.4f}")
    print(f"  Error rate      : {_safe_pct(metrics.get('error_rate'))}")
    print(f"  Run cost        : ${report['total_cost_usd']:.4f}  (answers ${report['answer_cost_usd']:.4f}"
          f" + judge ${report['judge_cost_usd']:.4f})")
    print(DIVIDER)

    if report["overall_result"] == "PASS":
        print("  OVERALL: PASS - Safe to deploy")
    else:
        print("  OVERALL: FAIL - Deployment blocked")
        print(f"  Failed gates : {', '.join(report['failed_gates'])}")

    print(DIVIDER)


# =============================================================================
# Main
# =============================================================================

def new_run_meta(args: argparse.Namespace) -> dict:
    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit_id":     _get_commit_id(),
        "full_run":      args.full,
    }


def main(args: argparse.Namespace, run_meta: dict) -> int:
    """
    Orchestrate the full evaluation pipeline.
    Returns 0 (PASS) or 1 (FAIL). Raises on errors that prevent a meaningful run.
    """
    run_start = time.perf_counter()
    eval_cfg  = load_config()["evaluation"]
    run_meta["models"] = {
        "answer":    eval_cfg["model"],
        "judge":     eval_cfg["judge_model"],
        "embedding": eval_cfg["embedding_model"],
    }

    # ── Step 1: Load dataset ───────────────────────────────────────────────────
    dataset = load_golden_dataset()
    run_meta["dataset_size"] = len(dataset)
    limit     = None if args.full else (args.limit or eval_cfg.get("test_mode_questions", 10))
    questions = select_questions(dataset, limit)

    DIVIDER = "=" * 44
    mode_label = f"FULL RUN — {len(questions)} questions" if args.full \
        else f"TEST MODE — {len(questions)} of {len(dataset)} questions"
    print(f"\n{DIVIDER}")
    print(f"  ASU LLM Evaluation Run [{mode_label}]")
    print(f"  Timestamp : {run_meta['run_timestamp']}")
    print(f"  Commit    : {run_meta['commit_id']}")
    print(DIVIDER)

    categories = sorted({q.get("category", "") for q in questions})
    print(f"\n[Step 1/5] Loaded {len(dataset)} questions; evaluating {len(questions)} "
          f"across {len(categories)} categories: {', '.join(categories)}")

    # ── Preflight + Step 2: RAG pipeline ───────────────────────────────────────
    run_preflight()

    rag_start = time.perf_counter()
    records   = run_rag_pipeline(questions)
    print(f"\n  RAG pipeline done — {len(records)} answers in {_format_duration(time.perf_counter() - rag_start)}.")

    # ── Step 3: RAGAS evaluation ──────────────────────────────────────────────
    eval_start  = time.perf_counter()
    judge_usage = run_evaluation(records)
    print(f"\n  RAGAS evaluation done in {_format_duration(time.perf_counter() - eval_start)}.")

    # ── Steps 4 + 5: Aggregate + quality gates ────────────────────────────────
    metrics      = build_metrics(records)
    gate_results = run_quality_gates(metrics)

    # ── Save + summarise ──────────────────────────────────────────────────────
    report = build_report(run_meta, records, metrics, gate_results, judge_usage)
    save_all(report, args.results_dir)
    print_final_summary(report, time.perf_counter() - run_start)

    return 0 if gate_results["overall"] == "PASS" else 1


def handle_crash(exc: Exception, run_meta: dict, results_dir: str) -> int:
    """
    Safety net — always write an ERROR report so the CI artifact upload,
    step summary and dashboards show what went wrong.
    """
    error = f"{type(exc).__name__}: {exc}"
    hint  = error_hint(error)
    tb    = traceback.format_exc()

    print(f"\n{'=' * 44}")
    print("  OVERALL: ERROR - the evaluation could not run")
    print(f"  Error : {error}")
    if hint:
        print(f"  Hint  : {hint}")
    print(f"{'=' * 44}")
    if not isinstance(exc, PipelineError):
        print(tb)

    save_all(build_error_report(run_meta, error, tb), results_dir)
    return 1


if __name__ == "__main__":
    # Never crash on characters the console encoding can't show (Windows cp1252)
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(errors="replace")

    _args     = parse_args()
    _run_meta = new_run_meta(_args)

    try:
        exit_code = main(_args, _run_meta)
    except Exception as _exc:
        exit_code = handle_crash(_exc, _run_meta, _args.results_dir)

    # os._exit() hard-exits without running Python's cleanup phase.
    # sys.exit() raises SystemExit, which triggers destructor calls on
    # C-extension objects (RAGAS, Pinecone gRPC, aiohttp) and can segfault
    # on Windows during shutdown — producing exit code 139 even when the
    # evaluation fully succeeded (exit_code=0). os._exit() bypasses all of
    # that and delivers the correct exit code directly to the OS.
    # Flush stdout/stderr first — os._exit() skips Python's normal I/O teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
