# =============================================================================
# src/run_eval.py
# ASU LLM Evaluation — Master Evaluation Controller
#
# This is the file GitHub Actions runs on every push.
# It wires together:
#   rag_pipeline  → answers all golden-dataset questions
#   evaluator     → scores each answer with RAGAS
#   quality_gates → aggregates metrics + runs PASS/FAIL gates
#   SQLite        → persists every run to results/eval_history.db
#   JSON          → writes results/latest_report.json for the dashboard
#
# Exit codes (critical for GitHub Actions):
#   0 = PASS  → safe to deploy
#   1 = FAIL  → deployment blocked
# =============================================================================

# -----------------------------------------------------------------------------
# TEST_MODE flag — flip to False for the full 100-question production run
# -----------------------------------------------------------------------------
TEST_MODE = True   # True = first 10 questions only

# =============================================================================

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 0. Path bootstrap — add src/ to sys.path so sibling modules are importable
#    whether this script is run as:
#      python src/run_eval.py       (from project root)
#      python run_eval.py           (from inside src/)
# -----------------------------------------------------------------------------

_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Load .env before any OpenAI-touching imports
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=_ENV_PATH)

# NOTE: rag_pipeline / evaluator / quality_gates are imported *inside* main()
# so that any import-time exception (e.g. missing OPENAI_API_KEY) is caught by
# the top-level handler in __main__ which writes a minimal error report.
# This guarantees results/latest_report.json always exists after a CI run.

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_GOLDEN_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "golden_dataset.json")
_RESULTS_DIR         = os.path.join(_PROJECT_ROOT, "results")
_DB_PATH             = os.path.join(_RESULTS_DIR, "eval_history.db")
_REPORT_PATH         = os.path.join(_RESULTS_DIR, "latest_report.json")

# Cost per token: approximate GPT-4o blended rate ($5/M tokens, as specified)
_COST_PER_TOKEN_USD  = 0.000005


# =============================================================================
# Utility helpers
# =============================================================================

def _format_duration(seconds: float) -> str:
    """Convert a raw second count to a readable string like '4m 32s'."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s" if m > 0 else f"{s}s"


def _get_commit_id() -> str:
    """
    Return the short git commit hash of HEAD.
    Falls back to 'local' if git is unavailable or the folder is not a repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "local"


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


# =============================================================================
# Step 1 — Load golden dataset
# =============================================================================

def load_golden_dataset() -> list:
    """
    Load questions + expected answers from data/golden_dataset.json.
    Applies TEST_MODE slice if enabled.
    """
    print("\n[Step 1/5] Loading golden dataset...")

    if not os.path.exists(_GOLDEN_DATASET_PATH):
        raise FileNotFoundError(
            f"Golden dataset not found at: {_GOLDEN_DATASET_PATH}\n"
            "Make sure data/golden_dataset.json is populated."
        )

    with open(_GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("golden_dataset.json is empty.")

    full_count = len(dataset)

    if TEST_MODE:
        dataset = dataset[:10]
        print(f"  *** RUNNING IN TEST MODE — 10 questions only (of {full_count}) ***")
    else:
        print(f"  Loaded {full_count} questions from golden_dataset.json")

    print(f"  Running evaluation on: {len(dataset)} question(s)")
    return dataset


# =============================================================================
# Step 2 — RAG pipeline (answer every question)
# =============================================================================

def run_rag_pipeline(dataset: list) -> list:
    """
    Pass every question through rag_pipeline.get_answer().
    Individual failures are caught, logged, and replaced with an error stub
    so the rest of the run is never aborted.
    """
    total   = len(dataset)
    results = []

    print(f"\n[Step 2/5] Running RAG pipeline ({total} question(s))...")

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        print(f"  Answering question {i}/{total}: {question[:70]}...")

        try:
            result = rag_pipeline.get_answer(question)
        except Exception as exc:
            # Log the failure but keep going
            print(f"  [ERROR] Question {i} failed in RAG pipeline: {exc}")
            result = {
                "question":         question,
                "answer":           f"[RAG ERROR] {exc}",
                "retrieved_chunks": [],
                "source_documents": [],
                "latency_seconds":  0.0,
                "token_usage":      0,
            }

        # Attach golden-dataset metadata for later reporting
        result["golden_id"]  = item.get("id")
        result["category"]   = item.get("category", "")
        result["difficulty"] = item.get("difficulty", "")
        results.append(result)

    return results


# =============================================================================
# Step 3 — RAGAS evaluation (score every answer)
# =============================================================================

def run_evaluation(rag_results: list, dataset: list) -> list:
    """
    Score every RAG result using evaluator.evaluate_batch().
    Ground-truth references are pulled from the golden dataset's
    expected_answer field so that ContextPrecision (with reference) is used.

    Falls back to individual evaluate_single() calls if the batch call
    raises an unexpected exception.
    """
    print(f"\n[Step 3/5] Scoring {len(rag_results)} answer(s) with RAGAS...")

    # Build the reference list in the same order as rag_results
    references = [item.get("expected_answer", "") for item in dataset]

    try:
        scored = evaluator.evaluate_batch(rag_results, references=references)

    except Exception as batch_exc:
        print(f"  [WARNING] Batch evaluation raised: {batch_exc}")
        print("  Falling back to question-by-question evaluation...")
        scored = []
        for i, (result, ref) in enumerate(zip(rag_results, references), start=1):
            print(f"  Evaluating question {i}/{len(rag_results)} (fallback)...")
            try:
                s = evaluator.evaluate_single(result, reference=ref)
            except Exception as single_exc:
                s = {
                    **result,
                    "faithfulness":      None,
                    "answer_relevancy":  None,
                    "context_precision": None,
                    "evaluation_error":  f"{type(single_exc).__name__}: {single_exc}",
                }
            scored.append(s)

    return scored


# =============================================================================
# Step 4 — Aggregate metrics
# =============================================================================

def build_metrics(scored_results: list) -> dict:
    """
    Delegate to quality_gates.compute_aggregate_metrics() and log the results.
    """
    print("\n[Step 4/5] Computing aggregate metrics...")
    metrics = quality_gates.compute_aggregate_metrics(scored_results)

    print("  Aggregated metrics:")
    for key, val in metrics.items():
        print(f"    {key:<24}: {_safe_val(val)}")

    return metrics


# =============================================================================
# Step 5 — Quality gates
# =============================================================================

def run_quality_gates(metrics: dict) -> dict:
    """Run quality gates and print the gate report."""
    print("\n[Step 5/5] Running quality gates...")
    gate_results = quality_gates.check_gates(metrics)
    quality_gates.print_gate_report(gate_results)
    return gate_results


# =============================================================================
# Save results — SQLite
# =============================================================================

def save_to_database(
    run_timestamp: str,
    commit_id: str,
    dataset: list,
    metrics: dict,
    gate_results: dict,
) -> None:
    """
    Persist one row per evaluation run to results/eval_history.db.
    Creates the database and table automatically on first run.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    conn = sqlite3.connect(_DB_PATH)
    try:
        # Create table if it doesn't exist yet
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp       TEXT,
                commit_id           TEXT,
                total_questions     INTEGER,
                hallucination_rate  REAL,
                answer_relevancy    REAL,
                faithfulness        REAL,
                context_precision   REAL,
                latency_p95_seconds REAL,
                cost_per_query_usd  REAL,
                overall_result      TEXT,
                failed_gates        TEXT
            )
        """)

        conn.execute(
            """
            INSERT INTO eval_runs (
                run_timestamp, commit_id, total_questions,
                hallucination_rate, answer_relevancy, faithfulness,
                context_precision, latency_p95_seconds, cost_per_query_usd,
                overall_result, failed_gates
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_timestamp,
                commit_id,
                len(dataset),
                metrics.get("hallucination_rate"),
                metrics.get("answer_relevancy"),
                metrics.get("faithfulness"),
                metrics.get("context_precision"),
                metrics.get("latency_p95_seconds"),
                metrics.get("cost_per_query_usd"),
                gate_results["overall"],
                ", ".join(gate_results.get("failed_gates", [])),
            ),
        )
        conn.commit()

        # Confirm row was written
        row = conn.execute(
            "SELECT id FROM eval_runs WHERE run_timestamp = ?",
            (run_timestamp,)
        ).fetchone()
        print(f"  [OK] eval_history.db — row inserted (id={row[0]}, "
              f"timestamp={run_timestamp})")

    finally:
        conn.close()


# =============================================================================
# Save results — JSON report
# =============================================================================

def save_json_report(
    run_timestamp: str,
    commit_id: str,
    dataset: list,
    metrics: dict,
    gate_results: dict,
    scored_results: list,
) -> None:
    """
    Write a full-detail JSON report to results/latest_report.json.
    This file is read by the Streamlit dashboard.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # Collect up to 5 sample failures (faithfulness < 0.5)
    sample_failures = []
    for r in scored_results:
        f = r.get("faithfulness")
        if f is not None and f < 0.5:
            sample_failures.append({
                "question":    r["question"],
                "answer":      r["answer"][:300],   # truncate long answers
                "faithfulness": f,
                "category":    r.get("category", ""),
                "error":       r.get("evaluation_error"),
            })
        if len(sample_failures) >= 5:
            break

    report = {
        "run_timestamp":   run_timestamp,
        "commit_id":       commit_id,
        "total_questions": len(dataset),
        "test_mode":       TEST_MODE,
        "metrics":         metrics,
        "gate_results": {
            "overall":      gate_results["overall"],
            "passed_gates": gate_results["passed_gates"],
            "failed_gates": gate_results["failed_gates"],
            "gates":        gate_results["gates"],
        },
        "overall_result":  gate_results["overall"],
        "failed_gates":    gate_results["failed_gates"],
        "sample_failures": sample_failures,
    }

    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  [OK] latest_report.json — saved to {_REPORT_PATH}")


# =============================================================================
# Final summary
# =============================================================================

def print_final_summary(
    dataset: list,
    elapsed: float,
    commit_id: str,
    metrics: dict,
    gate_results: dict,
) -> None:
    """Print the human-readable evaluation summary to stdout."""
    DIVIDER  = "=" * 44
    overall  = gate_results["overall"]
    duration = _format_duration(elapsed)

    print(f"\n{DIVIDER}")
    print("EVALUATION COMPLETE")
    print(DIVIDER)
    print(f"  Total questions : {len(dataset)}"
          + (" (TEST MODE)" if TEST_MODE else ""))
    print(f"  Time taken      : {duration}")
    print(f"  Commit          : {commit_id}")
    print(DIVIDER)
    print(f"  Hallucination   : {_safe_pct(metrics.get('hallucination_rate'))}")
    print(f"  Answer Relevancy: {_safe_val(metrics.get('answer_relevancy'))}")
    print(f"  Faithfulness    : {_safe_val(metrics.get('faithfulness'))}")
    print(f"  Context Prec    : {_safe_val(metrics.get('context_precision'))}")
    print(f"  Latency p95     : {_safe_val(metrics.get('latency_p95_seconds'))}s")
    print(f"  Cost per query  : ${metrics.get('cost_per_query_usd') or 0:.4f}")
    print(DIVIDER)

    if overall == "PASS":
        print("  OVERALL: PASS - Safe to deploy")
    else:
        failed = ", ".join(gate_results["failed_gates"])
        print("  OVERALL: FAIL - Deployment blocked")
        print(f"  Failed gates : {failed}")

    print(DIVIDER)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """
    Orchestrate the full evaluation pipeline.
    Returns 0 (PASS) or 1 (FAIL) for GitHub Actions.
    """
    # Pipeline module imports are here (not at file top) so that any import-time
    # error (missing API key, bad dep) is caught by the __main__ crash handler.
    # global declarations make the names visible to all helper functions in this
    # module (run_rag_pipeline, run_evaluation, build_metrics, etc.).
    global rag_pipeline, evaluator, quality_gates
    import rag_pipeline
    import evaluator
    import quality_gates

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    commit_id     = _get_commit_id()
    run_start     = time.perf_counter()

    # ── Header ────────────────────────────────────────────────────────────────
    DIVIDER = "=" * 44
    print(f"\n{DIVIDER}")
    mode_label = "TEST MODE — 10 questions" if TEST_MODE else "FULL RUN — 100 questions"
    print(f"  ASU LLM Evaluation Run [{mode_label}]")
    print(f"  Timestamp : {run_timestamp}")
    print(f"  Commit    : {commit_id}")
    print(DIVIDER)

    # ── Warmup — absorb cold-start cost before timing begins ─────────────────
    # The first call to rag_pipeline.get_answer() pays a one-time price:
    #   • loads all 6 .txt documents from disk
    #   • splits them into chunks
    #   • loads or builds the ChromaDB vectorstore
    # This can add 4–6 s to the first question, inflating the p95 latency.
    # Running one throwaway question here means every timed question hits a
    # warm vectorstore and the timing results are fair and reproducible.
    print("\n[Warmup] Warming up RAG pipeline (not counted in results)...")
    try:
        rag_pipeline.get_answer("warmup")
        print("[Warmup] Pipeline is warm. Starting timed evaluation.\n")
    except Exception as warmup_exc:
        print(f"[Warmup] Warning — warmup query failed: {warmup_exc}")
        print("[Warmup] Continuing anyway — first question may be slower.\n")

    # ── Step 1: Load dataset ───────────────────────────────────────────────────
    dataset = load_golden_dataset()

    # ── Step 2: RAG pipeline ──────────────────────────────────────────────────
    rag_start   = time.perf_counter()
    rag_results = run_rag_pipeline(dataset)
    rag_elapsed = time.perf_counter() - rag_start
    print(f"\n  RAG pipeline done — {len(rag_results)} answers in {_format_duration(rag_elapsed)}.")

    # ── Step 3: RAGAS evaluation ──────────────────────────────────────────────
    eval_start     = time.perf_counter()
    scored_results = run_evaluation(rag_results, dataset)
    eval_elapsed   = time.perf_counter() - eval_start
    print(f"\n  RAGAS evaluation done — {len(scored_results)} scored in {_format_duration(eval_elapsed)}.")

    # ── Step 4: Aggregate ─────────────────────────────────────────────────────
    metrics = build_metrics(scored_results)

    # ── Step 5: Quality gates ─────────────────────────────────────────────────
    gate_results = run_quality_gates(metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n[Saving results...]")
    try:
        save_to_database(run_timestamp, commit_id, dataset, metrics, gate_results)
    except Exception as exc:
        print(f"  [WARNING] Could not save to SQLite: {exc}")

    try:
        save_json_report(
            run_timestamp, commit_id, dataset,
            metrics, gate_results, scored_results
        )
    except Exception as exc:
        print(f"  [WARNING] Could not save JSON report: {exc}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - run_start
    print_final_summary(dataset, total_elapsed, commit_id, metrics, gate_results)

    return 0 if gate_results["overall"] == "PASS" else 1


if __name__ == "__main__":
    try:
        exit_code = main()

    except Exception as exc:
        # ----------------------------------------------------------------
        # Crash safety net — always write a minimal report so the CI
        # artifact upload step has a file to upload, even if the pipeline
        # crashes before reaching save_json_report().
        # ----------------------------------------------------------------
        import traceback as _tb

        _ts = datetime.now().isoformat(timespec="seconds")
        _err_str = f"{type(exc).__name__}: {exc}"
        _tb_str  = _tb.format_exc()

        print(f"\n{'=' * 44}")
        print(f"  [FATAL] Pipeline crashed before completing.")
        print(f"  Error  : {_err_str}")
        print(f"{'=' * 44}")
        print(_tb_str)

        # Try to write a minimal error report so the dashboard and CI
        # artifact always have something to show.
        try:
            os.makedirs(_RESULTS_DIR, exist_ok=True)
            _error_report = {
                "run_timestamp":   _ts,
                "commit_id":       _get_commit_id(),
                "total_questions": 0,
                "test_mode":       TEST_MODE,
                "overall_result":  "ERROR",
                "failed_gates":    [],
                "metrics":         {},
                "gate_results": {
                    "overall":      "ERROR",
                    "passed_gates": [],
                    "failed_gates": [],
                    "gates":        {},
                },
                "sample_failures": [],
                "pipeline_error":  _err_str,
                "traceback":       _tb_str,
            }
            with open(_REPORT_PATH, "w", encoding="utf-8") as _f:
                json.dump(_error_report, _f, indent=2)
            print(f"[FATAL] Error report saved → {_REPORT_PATH}")
        except Exception as _report_exc:
            print(f"[FATAL] Could not write error report: {_report_exc}")

        exit_code = 1

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
