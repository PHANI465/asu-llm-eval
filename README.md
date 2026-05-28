# ASU LLM Evaluation Pipeline

An automated CI/CD pipeline that evaluates the quality of an ASU university RAG chatbot on every GitHub push. Uses **RAGAS** to score faithfulness, relevancy, and precision; enforces thresholds through quality gates; and visualises results in a live Streamlit dashboard.

---

## Architecture

```
GitHub Push
     |
     v
GitHub Actions (eval.yml)
     |
     +---> src/run_eval.py  (master orchestrator)
               |
               +---> src/rag_pipeline.py   ChromaDB + GPT-4o answers
               |
               +---> src/evaluator.py      RAGAS scoring
               |
               +---> src/quality_gates.py  PASS / FAIL gates
               |
               +---> results/eval_history.db    (SQLite — run history)
               +---> results/latest_report.json (dashboard data)
               |
               v
          Exit 0 (PASS) or Exit 1 (FAIL)
               |
               v
     dashboard/app.py  (Streamlit — view locally)
```

---

## Quality Gates

| Gate | Threshold | Direction |
|---|---|---|
| Hallucination Rate | <= 0.05 | lower is better |
| Answer Relevancy | >= 0.75 | higher is better |
| Faithfulness | >= 0.80 | higher is better |
| Context Precision | >= 0.60 | higher is better |
| Latency P95 | <= 3.0 s | lower is better |
| Cost Per Query | <= $0.02 | lower is better |

All thresholds are configurable in `config.yaml`.

---

## Tech Stack

| Component | Library / Service |
|---|---|
| LLM (answers + judge) | OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| Vector store | ChromaDB 1.0.7 |
| RAG framework | LangChain 0.3.25 |
| Evaluation | RAGAS 0.2.15 |
| Dashboard | Streamlit 1.45.1 |
| Run history | SQLite (stdlib) |
| CI/CD | GitHub Actions |

---

## Project Structure

```
asu-llm-eval/
├── .github/
│   └── workflows/
│       └── eval.yml              # CI/CD pipeline
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── data/
│   ├── golden_dataset.json       # 100 Q&A pairs for evaluation
│   └── knowledge_base/
│       ├── undergraduate_admissions.txt
│       ├── graduate_admissions.txt
│       ├── tuition_and_fees.txt
│       ├── housing_and_dining.txt
│       ├── scholarships_and_financial_aid.txt
│       └── campus_and_programs.txt
├── results/
│   ├── eval_history.db           # SQLite run history (gitignored)
│   └── latest_report.json        # Dashboard data (gitignored)
├── src/
│   ├── rag_pipeline.py           # RAG core: load, chunk, embed, answer
│   ├── evaluator.py              # RAGAS scoring engine
│   ├── quality_gates.py          # Threshold checks + PASS/FAIL
│   └── run_eval.py               # Master orchestrator (CI entry point)
├── chroma_db/                    # Persisted vector store (gitignored)
├── config.yaml                   # All thresholds and evaluation settings
├── requirements.txt
├── .env                          # OPENAI_API_KEY (gitignored — never commit)
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/asu-llm-eval.git
cd asu-llm-eval
pip install -r requirements.txt
```

### 2. Add your API key

```bash
# Create .env in the project root
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Run the evaluation (TEST_MODE — 10 questions)

```bash
python src/run_eval.py
```

Output:
```
ASU LLM Evaluation Run [TEST MODE -- 10 questions]
...
QUALITY GATE REPORT
============================================
  hallucination_rate  : PASS  (0.0000 <= 0.05)
  answer_relevancy    : PASS  (0.9304 >= 0.75)
  faithfulness        : PASS  (1.0000 >= 0.8)
  context_precision   : PASS  (0.9333 >= 0.6)
  latency_p95         : PASS  (1.902s <= 3.0s)
  cost_per_query      : PASS  ($0.0056 <= $0.02)
============================================
OVERALL RESULT: PASS
All 6 gates passed. Safe to deploy.
```

### 4. Run the full evaluation (100 questions)

In `src/run_eval.py`, change:
```python
TEST_MODE = False
```
Then re-run `python src/run_eval.py`.

### 5. View the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` and shows:
- Current gate status (color-coded PASS/FAIL)
- Key metric cards
- Historical trend charts
- Sample failure details
- Full run history table

---

## CI/CD Setup (GitHub Actions)

### 1. Add the secret

In your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | `sk-...` |

### 2. Push to trigger

```bash
git add .
git commit -m "feat: add new knowledge base doc"
git push origin main
```

The workflow in `.github/workflows/eval.yml` runs automatically on every push to `main`/`master` and on pull requests.

### 3. Manual full run

Go to **Actions → LLM Evaluation Pipeline → Run workflow** and select `full_run = true` to run all 100 questions on demand.

### 4. Download the report

After any run, go to **Actions → your run → Artifacts** and download `evaluation-report-<run_number>` to inspect `latest_report.json`.

---

## Configuration Reference

All settings live in `config.yaml`:

```yaml
quality_gates:
  hallucination_rate_max:   0.05   # max tolerable hallucination rate
  answer_relevancy_min:     0.75   # min RAGAS answer relevancy score
  faithfulness_min:         0.80   # min RAGAS faithfulness score
  context_precision_min:    0.60   # min RAGAS context precision score
  latency_p95_max_seconds:  3.0    # max 95th-percentile latency in seconds
  cost_per_query_max_usd:   0.02   # max estimated cost per query

evaluation:
  model:            gpt-4o                 # LLM for answers and RAGAS judge
  embedding_model:  text-embedding-3-small # embedding model for ChromaDB
  chunk_size:       1000                   # characters per chunk
  chunk_overlap:    150                    # overlap between adjacent chunks
  top_k_retrieval:  5                      # chunks retrieved per question
```

---

## Module API Reference

### `src/rag_pipeline.py`

```python
get_answer(question: str) -> dict
# Returns:
# {
#   "question":         str,
#   "answer":           str,
#   "retrieved_chunks": list[str],
#   "source_documents": list[str],
#   "latency_seconds":  float,
#   "token_usage":      int,
# }
```

### `src/evaluator.py`

```python
evaluate_single(result: dict, reference: str = "") -> dict
evaluate_batch(results: list, references: list = None) -> list
# Each scored dict adds: faithfulness, answer_relevancy, context_precision, evaluation_error
```

### `src/quality_gates.py`

```python
compute_hallucination_rate(results: list) -> float
compute_aggregate_metrics(results: list) -> dict
check_gates(metrics: dict) -> dict   # returns {"overall": "PASS"|"FAIL", "gates": {...}, ...}
print_gate_report(gate_results: dict) -> None
```

---

## Extending the Pipeline

### Add a new knowledge base document

1. Drop a `.txt` file into `data/knowledge_base/`
2. Delete `chroma_db/` so it is rebuilt with the new document
3. Add relevant Q&A pairs to `data/golden_dataset.json`
4. Push to `main` — the CI pipeline picks it up automatically

### Add a new quality gate

1. Add the threshold to `config.yaml` under `quality_gates`
2. Add a gate definition tuple to `gate_definitions` in `src/quality_gates.py`
3. Ensure the metric is computed in `compute_aggregate_metrics()`

### Adjust thresholds

Edit `config.yaml` — no code changes required. The gate engine reads thresholds at runtime.

---

## Sample Evaluation Results

Results from the last TEST_MODE run (10 questions):

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Hallucination Rate | 0.0000 | <= 0.05 | PASS |
| Answer Relevancy | 0.9304 | >= 0.75 | PASS |
| Faithfulness | 1.0000 | >= 0.80 | PASS |
| Context Precision | 0.9333 | >= 0.60 | PASS |
| Latency P95 | 1.902 s | <= 3.0 s | PASS |
| Cost Per Query | $0.0056 | <= $0.02 | PASS |
