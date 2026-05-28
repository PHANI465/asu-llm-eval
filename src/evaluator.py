# =============================================================================
# src/evaluator.py
# ASU LLM Evaluation — RAGAS Scoring Engine
#
# Responsibilities:
#   1. Wrap each RAG result into a RAGAS EvaluationDataset (SingleTurnSample)
#   2. Score with three metrics:
#        - Faithfulness                         (no reference required)
#        - AnswerRelevancy                      (no reference required)
#        - ContextPrecision / LLMContextPrecisionWithoutReference
#            -> uses ContextPrecision when a ground-truth reference is given
#            -> falls back to LLMContextPrecisionWithoutReference otherwise
#   3. Pass through latency_seconds + token_usage from the RAG result dict
#   4. Return scored dicts; batch mode never aborts on a single failure
# =============================================================================

import math
import os
import sys

from dotenv import load_dotenv
from tabulate import tabulate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,                      # requires reference (ground truth)
    LLMContextPrecisionWithoutReference,   # no reference needed
)

# -----------------------------------------------------------------------------
# 0. Load env vars
# -----------------------------------------------------------------------------

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. "
        "Make sure it is set in the .env file at the project root."
    )

# Judge model constants (same models as the RAG pipeline for consistency)
_LLM_MODEL   = "gpt-4o"
_EMBED_MODEL = "text-embedding-3-small"

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
        print(f"            LLM        : {_LLM_MODEL}")
        print(f"            Embeddings : {_EMBED_MODEL}")

        _llm_judge = LangchainLLMWrapper(
            ChatOpenAI(
                model=_LLM_MODEL,
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
            )
        )
        _embed_judge = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=_EMBED_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )
        )
        print("[Evaluator] RAGAS judge ready.\n")

    return _llm_judge, _embed_judge


# -----------------------------------------------------------------------------
# 2. Helper — safe float conversion (handles NaN from RAGAS)
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


# -----------------------------------------------------------------------------
# 3. evaluate_single — score one RAG result
# -----------------------------------------------------------------------------

def evaluate_single(result: dict, reference: str = "") -> dict:
    """
    Score a single RAG result dict using RAGAS.

    Parameters
    ----------
    result    : dict returned by rag_pipeline.get_answer()
                Must contain: question, answer, retrieved_chunks,
                              latency_seconds, token_usage
    reference : optional ground-truth answer string.
                When provided, ContextPrecision (with reference) is used.
                When omitted, LLMContextPrecisionWithoutReference is used.

    Returns
    -------
    dict with keys:
        question, answer,
        faithfulness, answer_relevancy, context_precision,
        latency_seconds, token_usage, evaluation_error
    """
    question = result["question"]
    answer   = result["answer"]
    contexts = result.get("retrieved_chunks") or []   # list[str]

    # Prepare output with safe defaults
    output = {
        "question":          question,
        "answer":            answer,
        "faithfulness":      None,
        "answer_relevancy":  None,
        "context_precision": None,
        "latency_seconds":   result.get("latency_seconds", 0),
        "token_usage":       result.get("token_usage", 0),
        "evaluation_error":  None,
    }

    try:
        llm_judge, embed_judge = _ensure_judge()

        # ----- Build a single RAGAS sample --------------------------------
        # SingleTurnSample uses the new ragas 0.2.x field names:
        #   user_input        = the question
        #   response          = the generated answer
        #   retrieved_contexts = list of chunk strings used by the RAG
        #   reference         = ground-truth answer (optional)
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference if reference else None,
        )

        dataset = EvaluationDataset(samples=[sample])

        # ----- Choose context-precision metric based on reference ----------
        # With reference    -> ContextPrecision     (output col: "context_precision")
        # Without reference -> LLMContextPrecisionWithoutReference
        #                      (output col: "llm_context_precision_without_reference")
        if reference:
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

        # ----- Run RAGAS evaluation ---------------------------------------
        # raise_exceptions=False: individual metric failures return NaN
        # instead of crashing the whole evaluation.
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_judge,
            embeddings=embed_judge,
            raise_exceptions=False,
            show_progress=False,    # suppress ragas internal progress bar
        )

        # ----- Extract scores from the result DataFrame -------------------
        df  = eval_result.to_pandas()
        row = df.iloc[0]

        output["faithfulness"]      = _safe_float(row.get("faithfulness"))
        output["answer_relevancy"]  = _safe_float(row.get("answer_relevancy"))
        # Always store under "context_precision" regardless of which variant ran
        output["context_precision"] = _safe_float(row.get(ctx_precision_col))

    except Exception as exc:
        # Record the error but do NOT re-raise — caller decides what to do
        output["evaluation_error"] = f"{type(exc).__name__}: {exc}"

    return output


# -----------------------------------------------------------------------------
# 4. evaluate_batch — score a list of RAG results
# -----------------------------------------------------------------------------

def evaluate_batch(results: list, references: list = None) -> list:
    """
    Score every item in a list of RAG result dicts.

    Parameters
    ----------
    results    : list of dicts from rag_pipeline.get_answer()
    references : optional list of ground-truth strings, same length as results.
                 Pass None or omit for no-reference scoring.

    Returns
    -------
    list of scored dicts in the same order as input.
    Individual failures are captured in 'evaluation_error'; the batch continues.
    """
    if references is None:
        references = [""] * len(results)

    scored = []
    total  = len(results)

    for idx, (result, ref) in enumerate(zip(results, references), start=1):
        q_preview = result["question"][:72]
        print(f"Evaluating question {idx}/{total}: \"{q_preview}\"")

        scored_result = evaluate_single(result, reference=ref)

        if scored_result["evaluation_error"]:
            print(f"  [ERROR] {scored_result['evaluation_error']}")
        else:
            f  = scored_result["faithfulness"]
            ar = scored_result["answer_relevancy"]
            cp = scored_result["context_precision"]
            print(
                f"  Question {idx} scored: "
                f"faithfulness={f}  "
                f"answer_relevancy={ar}  "
                f"context_precision={cp}"
            )

        scored.append(scored_result)

    return scored


# -----------------------------------------------------------------------------
# 5. Quick-test entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure src/ is on the path so we can import sibling modules
    _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)

    import rag_pipeline

    print("=" * 62)
    print("  ASU LLM Evaluator - Quick Test")
    print("=" * 62)

    TEST_QUESTIONS = [
        "What is the minimum GPA for undergraduate admission?",
        "What is the tuition for international students?",
        "What meal plans are available at ASU?",
    ]

    # ------------------------------------------------------------------
    # Step 1: Get RAG answers
    # ------------------------------------------------------------------
    print("\n[Step 1/3] Querying RAG pipeline...")
    rag_results = []
    for q in TEST_QUESTIONS:
        print(f"  >> {q}")
        rag_results.append(rag_pipeline.get_answer(q))
    print(f"  >> {len(rag_results)} answers retrieved.\n")

    # ------------------------------------------------------------------
    # Step 2: Score with RAGAS (no reference — using LLMContextPrecisionWithoutReference)
    # ------------------------------------------------------------------
    print("[Step 2/3] Running RAGAS evaluations...")
    scored_results = evaluate_batch(rag_results)

    # ------------------------------------------------------------------
    # Step 3: Print summary table
    # ------------------------------------------------------------------
    print("\n[Step 3/3] Results summary")
    print("=" * 62)

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

    # Print full error detail if any
    errors = [(r["question"], r["evaluation_error"]) for r in scored_results if r["evaluation_error"]]
    if errors:
        print("\n[Full error details]")
        for q, e in errors:
            print(f"  Q: {q}\n  E: {e}\n")

    print("\nEvaluation complete.")
