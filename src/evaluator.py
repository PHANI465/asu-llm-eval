# =============================================================================
# src/evaluator.py
# ASU LLM Evaluation — RAGAS Scoring Engine
#
# Responsibilities:
#   1. Wrap RAG results into a RAGAS EvaluationDataset (SingleTurnSample)
#   2. Score with three metrics:
#        - Faithfulness                         (no reference required)
#        - AnswerRelevancy                      (no reference required)
#        - ContextPrecision / LLMContextPrecisionWithoutReference
#            -> uses ContextPrecision when a ground-truth reference is given
#            -> falls back to LLMContextPrecisionWithoutReference otherwise
#   3. Score the whole batch in one RAGAS run, so judge calls run concurrently
#   4. Skip results whose RAG call failed — an error message is not an answer
#   5. Track judge token usage + cost; one bad sample never aborts the batch
# =============================================================================

import math
import sys

from tabulate import tabulate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.cost import get_token_usage_for_openai
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,                      # requires reference (ground truth)
    LLMContextPrecisionWithoutReference,   # no reference needed
)
from ragas.run_config import RunConfig

from project_config import load_config, require_env, token_cost_usd

# -----------------------------------------------------------------------------
# 0. Settings from config.yaml (shared with the pipeline)
# -----------------------------------------------------------------------------

_EVAL_CFG     = load_config()["evaluation"]
JUDGE_MODEL   = _EVAL_CFG["judge_model"]                  # gpt-4o-mini  (cheaper judge)
EMBED_MODEL   = _EVAL_CFG["embedding_model"]              # text-embedding-3-small
JUDGE_WORKERS = _EVAL_CFG.get("judge_max_workers", 8)     # concurrent judge requests

# -----------------------------------------------------------------------------
# 1. Lazy singleton — judge LLM + embeddings (initialised once per process)
# -----------------------------------------------------------------------------

_llm_judge   = None
_embed_judge = None


def _ensure_judge():
    """
    Initialise the RAGAS judge on first call and reuse on subsequent calls.
    Printing here is intentional — it signals when the cold-start happens.
    """
    global _llm_judge, _embed_judge

    if _llm_judge is None:
        print("\n[Evaluator] Initialising RAGAS judge...")
        print(f"            LLM        : {JUDGE_MODEL}")
        print(f"            Embeddings : {EMBED_MODEL}")

        api_key = require_env("OPENAI_API_KEY")
        _llm_judge = LangchainLLMWrapper(
            ChatOpenAI(model=JUDGE_MODEL, temperature=0, openai_api_key=api_key)
        )
        _embed_judge = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)
        )
        print("[Evaluator] RAGAS judge ready.\n")

    return _llm_judge, _embed_judge


# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    """
    Convert a RAGAS metric score to a plain Python float.
    Returns None for NaN, None, or anything that can't be cast.
    """
    try:
        f = float(value)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _empty_score(result: dict) -> dict:
    return {
        "question":          result["question"],
        "answer":            result.get("answer", ""),
        "faithfulness":      None,
        "answer_relevancy":  None,
        "context_precision": None,
        "latency_seconds":   result.get("latency_seconds", 0),
        "token_usage":       result.get("token_usage", 0),
        "evaluation_error":  None,
    }


def _sum_token_usage(usage) -> tuple[int, int]:
    """RAGAS returns one TokenUsage, or a list when several models were called."""
    items = usage if isinstance(usage, list) else [usage]
    return (
        sum(u.input_tokens for u in items),
        sum(u.output_tokens for u in items),
    )


# -----------------------------------------------------------------------------
# 3. Batch scoring
# -----------------------------------------------------------------------------

def _score_group(indices, results, references, scored, use_reference, usage) -> None:
    """Score results[indices] in a single RAGAS run and write into scored[i]."""
    llm_judge, embed_judge = _ensure_judge()

    samples = [
        SingleTurnSample(
            user_input=results[i]["question"],
            response=results[i]["answer"],
            retrieved_contexts=results[i].get("retrieved_chunks") or [],
            reference=references[i] if use_reference else None,
        )
        for i in indices
    ]

    # With reference    -> ContextPrecision     (output col: "context_precision")
    # Without reference -> LLMContextPrecisionWithoutReference
    #                      (output col: "llm_context_precision_without_reference")
    if use_reference:
        ctx_precision_metric = ContextPrecision(llm=llm_judge)
        ctx_precision_col    = "context_precision"
    else:
        ctx_precision_metric = LLMContextPrecisionWithoutReference(llm=llm_judge)
        ctx_precision_col    = "llm_context_precision_without_reference"

    metrics = [
        Faithfulness(llm=llm_judge),
        AnswerRelevancy(llm=llm_judge, embeddings=embed_judge),
        ctx_precision_metric,
    ]

    print(f"  Scoring {len(indices)} answer(s) "
          f"{'with' if use_reference else 'without'} reference "
          f"({JUDGE_WORKERS} concurrent judge requests)...")

    try:
        # raise_exceptions=False: individual metric failures return NaN
        # instead of crashing the whole evaluation.
        eval_result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=llm_judge,
            embeddings=embed_judge,
            run_config=RunConfig(max_workers=JUDGE_WORKERS),
            token_usage_parser=get_token_usage_for_openai,
            raise_exceptions=False,
            show_progress=False,
        )
    except Exception as exc:
        for i in indices:
            scored[i]["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        return

    df = eval_result.to_pandas()
    for row_idx, i in enumerate(indices):
        row = df.iloc[row_idx]
        scored[i]["faithfulness"]      = _safe_float(row.get("faithfulness"))
        scored[i]["answer_relevancy"]  = _safe_float(row.get("answer_relevancy"))
        # Always store under "context_precision" regardless of which variant ran
        scored[i]["context_precision"] = _safe_float(row.get(ctx_precision_col))

        if all(scored[i][k] is None for k in ("faithfulness", "answer_relevancy", "context_precision")):
            scored[i]["evaluation_error"] = "RAGAS returned no scores (all judge calls failed)"

    try:
        input_tokens, output_tokens = _sum_token_usage(eval_result.total_tokens())
        usage["input_tokens"]  += input_tokens
        usage["output_tokens"] += output_tokens
    except Exception as exc:
        print(f"  [WARNING] Could not read judge token usage: {exc}")


def evaluate_batch(results: list, references: list = None) -> tuple[list, dict]:
    """
    Score every item in a list of RAG result dicts.

    Parameters
    ----------
    results    : list of dicts from rag_pipeline.get_answer()
    references : optional list of ground-truth strings, same length as results.
                 Pass None or omit for no-reference scoring.

    Returns
    -------
    (scored, judge_usage)
      scored      : list of score dicts in the same order as input, with keys
                    question, answer, faithfulness, answer_relevancy,
                    context_precision, latency_seconds, token_usage,
                    evaluation_error
      judge_usage : {"input_tokens", "output_tokens", "cost_usd"} for the judge

    Results whose RAG call failed (non-empty "error") are not sent to the judge;
    their scores stay None and evaluation_error explains why.
    """
    if references is None:
        references = [""] * len(results)
    if len(references) != len(results):
        raise ValueError(f"Got {len(references)} references for {len(results)} results.")

    scored = [_empty_score(r) for r in results]
    usage  = {"input_tokens": 0, "output_tokens": 0}

    to_score = []
    for i, r in enumerate(results):
        if r.get("error"):
            scored[i]["evaluation_error"] = f"Not scored — RAG call failed: {r['error']}"
        else:
            to_score.append(i)

    # One RAGAS run per metric set: context precision differs with/without reference
    with_reference    = [i for i in to_score if references[i]]
    without_reference = [i for i in to_score if not references[i]]
    for indices, use_reference in ((with_reference, True), (without_reference, False)):
        if indices:
            _score_group(indices, results, references, scored, use_reference, usage)

    for idx, s in enumerate(scored, start=1):
        if s["evaluation_error"]:
            print(f"  Question {idx}: [ERROR] {s['evaluation_error'][:160]}")
        else:
            print(f"  Question {idx}: faithfulness={s['faithfulness']}  "
                  f"answer_relevancy={s['answer_relevancy']}  "
                  f"context_precision={s['context_precision']}")

    usage["cost_usd"] = token_cost_usd(JUDGE_MODEL, usage["input_tokens"], usage["output_tokens"])
    return scored, usage


def evaluate_single(result: dict, reference: str = "") -> dict:
    """
    Score a single RAG result dict using RAGAS.

    Parameters
    ----------
    result    : dict returned by rag_pipeline.get_answer()
    reference : optional ground-truth answer string.
                When provided, ContextPrecision (with reference) is used.
                When omitted, LLMContextPrecisionWithoutReference is used.

    Returns
    -------
    score dict — see evaluate_batch()
    """
    scored, _ = evaluate_batch([result], [reference])
    return scored[0]


# -----------------------------------------------------------------------------
# 4. Quick-test entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import rag_pipeline

    print("=" * 62)
    print("  ASU LLM Evaluator - Quick Test")
    print("=" * 62)

    TEST_QUESTIONS = [
        "What is the minimum GPA for undergraduate admission?",
        "What is the tuition for international students?",
        "What meal plans are available at ASU?",
    ]

    print("\n[Step 1/3] Querying RAG pipeline...")
    rag_results = [rag_pipeline.get_answer(q) for q in TEST_QUESTIONS]
    print(f"  >> {len(rag_results)} answers retrieved.\n")

    print("[Step 2/3] Running RAGAS evaluations (no reference)...")
    scored_results, judge_usage = evaluate_batch(rag_results)

    print("\n[Step 3/3] Results summary")
    table_rows = []
    for r in scored_results:
        q_short = (r["question"][:42] + "...") if len(r["question"]) > 45 else r["question"]
        err = r["evaluation_error"]
        table_rows.append([
            q_short,
            r["faithfulness"]      if r["faithfulness"]      is not None else "N/A",
            r["answer_relevancy"]  if r["answer_relevancy"]  is not None else "N/A",
            r["context_precision"] if r["context_precision"] is not None else "N/A",
            f"{r['latency_seconds']}s",
            r["token_usage"],
            (err[:30] + "...") if err and len(err) > 33 else (err or "None"),
        ])

    headers = ["Question", "Faithful", "Ans Relev", "Ctx Prec", "Latency", "Tokens", "Error"]
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))

    answer_cost = sum(r.get("cost_usd") or 0 for r in rag_results)
    print(f"\n  Answer model ({rag_pipeline.LLM_MODEL}) cost : ${answer_cost:.5f}")
    print(f"  Judge ({JUDGE_MODEL}) tokens : {judge_usage['input_tokens']:,} in / "
          f"{judge_usage['output_tokens']:,} out  (${judge_usage['cost_usd'] or 0:.5f})")
    print("\nEvaluation complete.")

    # Hard exit avoids RAGAS / gRPC C-extension segfaults during interpreter cleanup on Windows
    sys.stdout.flush()
    os._exit(0)
