"""
Offline tests for rag_pipeline / evaluator / report_summary behaviour that
needs no API calls. Importing these modules loads LangChain + RAGAS, which
takes a few seconds but makes no network requests.
"""

from langchain_core.documents import Document

import evaluator
import rag_pipeline
import report_summary


def test_knowledge_base_fingerprint_tracks_content():
    chunks = [Document(page_content="Tuition is $12,177", metadata={"source": "tuition.txt"})]
    same   = [Document(page_content="Tuition is $12,177", metadata={"source": "tuition.txt"})]
    edited = [Document(page_content="Tuition is $12,500", metadata={"source": "tuition.txt"})]
    moved  = [Document(page_content="Tuition is $12,177", metadata={"source": "fees.txt"})]

    fingerprint = rag_pipeline.knowledge_base_fingerprint(chunks)
    assert fingerprint == rag_pipeline.knowledge_base_fingerprint(same)
    assert fingerprint != rag_pipeline.knowledge_base_fingerprint(edited)
    assert fingerprint != rag_pipeline.knowledge_base_fingerprint(moved)


def test_real_knowledge_base_loads_and_chunks():
    chunks = rag_pipeline.split_documents(rag_pipeline.load_documents())
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".txt") for chunk in chunks)


def test_get_answer_reports_failures_in_error_field(monkeypatch):
    def broken_vectorstore():
        raise RuntimeError("Pinecone unreachable")

    monkeypatch.setattr(rag_pipeline, "_get_vectorstore", broken_vectorstore)
    result = rag_pipeline.get_answer("What is tuition?")

    assert result["error"] == "RuntimeError: Pinecone unreachable"
    assert result["answer"] == ""          # an error message is never passed off as an answer
    assert result["cost_usd"] is None


def test_evaluator_does_not_send_failed_answers_to_the_judge(monkeypatch):
    def judge_must_not_be_called():
        raise AssertionError("judge was called for a failed RAG result")

    monkeypatch.setattr(evaluator, "_ensure_judge", judge_must_not_be_called)
    failed = {"question": "q", "answer": "", "retrieved_chunks": [], "latency_seconds": 1.0,
              "token_usage": 0, "error": "RateLimitError: 429"}

    scored, usage = evaluator.evaluate_batch([failed, dict(failed)], ["ref", ""])

    assert all(s["faithfulness"] is None for s in scored)
    assert all("RAG call failed" in s["evaluation_error"] for s in scored)
    assert usage["input_tokens"] == 0


def test_step_summary_markdown_for_error_and_pass():
    error_md = report_summary.build_markdown({
        "overall_result": "ERROR", "pipeline_error": "RateLimitError: 429", "error_hint": "Add credits",
    })
    assert "ERROR" in error_md and "Add credits" in error_md

    pass_md = report_summary.build_markdown({
        "overall_result": "PASS", "total_questions": 10, "dataset_size": 100, "test_mode": True,
        "commit_id": "abc1234", "models": {"answer": "gpt-4o", "judge": "gpt-4o-mini"},
        "total_cost_usd": 0.05,
        "gate_results": {"gates": {
            "error_rate":  {"value": 0.0, "threshold": 0.1, "direction": "max", "passed": True},
            "latency_p95": {"value": 4.2, "threshold": 15.0, "direction": "max", "passed": True},
        }},
    })
    assert "| Error rate | 0.0% | ≤ 10.0% | ✅ pass |" in pass_md
    assert "| Latency p95 | 4.20s | ≤ 15.00s | ✅ pass |" in pass_md
