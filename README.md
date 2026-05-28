# ASU LLM Evaluation Pipeline

An automated CI/CD pipeline that evaluates the quality of an ASU university RAG chatbot on every GitHub push. Uses **RAGAS** to score faithfulness, relevancy, and precision; enforces thresholds through quality gates; and visualises results in a live React dashboard deployed on Vercel.

> **Live Dashboard:** https://your-vercel-url.vercel.app
> **GitHub Repo:** https://github.com/PHANI465/asu-llm-eval

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
               +---> src/rag_pipeline.py
               |         Pinecone (cloud vectors)
               |         + GPT-4o answers
               |
               +---> src/evaluator.py
               |         RAGAS scoring
               |         GPT-4o-mini judge
               |
               +---> src/quality_gates.py
               |         PASS / FAIL gates
               |
               +---> results/latest_report.json
               |     (committed back to repo)
               |
               v
          Exit 0 (PASS) or Exit 1 (FAIL)
               |
               v
     React Dashboard on Vercel
     (auto-refreshes from GitHub)
```

---

## Quality Gates

| Gate | Threshold | Direction |
|---|---|---|
| Hallucination Rate | <= 0.10 | lower is better |
| Answer Relevancy | >= 0.75 | higher is better |
| Faithfulness | >= 0.80 | higher is better |
| Context Precision | >= 0.60 | higher is better |
| Latency P95 | <= 15.0 s | lower is better |
| Cost Per Query | <= $0.02 | lower is better |

All thresholds are configurable in `config.yaml`.

---

## Tech Stack

| Component | Library / Service |
|---|---|
| Answer LLM | GPT-4o |
| Judge LLM | GPT-4o-mini (cost optimised) |
| Embeddings | text-embedding-3-small |
| Vector store | Pinecone (cloud) |
| RAG framework | LangChain 0.3.25 |
| Evaluation | RAGAS 0.2.15 |
| Dashboard | React + Vite (Vercel) |
| Local dashboard | Streamlit 1.45.1 |
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
│   └── app.py                    # Streamlit dashboard (local)
├── dashboard-react/
│   ├── src/
│   │   └── App.jsx               # React dashboard
│   ├── vercel.json               # Vercel config
│   └── package.json
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
│   └── latest_report.json        # Dashboard data (committed by CI)
├── src/
│   ├── rag_pipeline.py           # RAG core: load, chunk, embed, answer
│   ├── evaluator.py              # RAGAS scoring engine
│   ├── quality_gates.py          # Threshold checks + PASS/FAIL
│   └── run_eval.py               # Master orchestrator (CI entry point)
├── (Pinecone used for cloud vector storage)
├── config.yaml                   # All thresholds and evaluation settings
├── requirements.txt
├── .env                          # API keys (gitignored — never commit)
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/PHANI465/asu-llm-eval.git
cd asu-llm-eval
pip install -r requirements.txt
```

### 2. Add your API keys

Create `.env` in the project root:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
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
  hallucination_rate  : PASS  (0.1000 <= 0.10)
  answer_relevancy    : PASS  (0.9284 >= 0.75)
  faithfulness        : PASS  (0.9000 >= 0.80)
  context_precision   : PASS  (0.9333 >= 0.60)
  latency_p95         : PASS  (2.169s <= 15.0s)
  cost_per_query      : PASS  ($0.0087 <= $0.02)
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

### 5. View the local Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## CI/CD Setup (GitHub Actions)

### 1. Add the secrets

In your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | `sk-...` |
| `PINECONE_API_KEY` | `pcsk-...` |

### 2. Push to trigger

```bash
git add .
git commit -m "feat: add new knowledge base doc"
git push origin main
```

The workflow in `.github/workflows/eval.yml` runs automatically on every push to `main`/`master` and on pull requests. After the run completes, `results/latest_report.json` is automatically committed back to the repo and the Vercel dashboard updates.

### 3. Manual full run

Go to **Actions → LLM Evaluation Pipeline → Run workflow** and select `full_run = true` to run all 100 questions on demand.

### 4. Download the report

After any run, go to **Actions → your run → Artifacts** and download `evaluation-report-<run_number>` to inspect `latest_report.json`.

---

## Configuration Reference

All settings live in `config.yaml`:

```yaml
quality_gates:
  hallucination_rate_max:   0.10   # max tolerable hallucination rate
  answer_relevancy_min:     0.75   # min RAGAS answer relevancy score
  faithfulness_min:         0.80   # min RAGAS faithfulness score
  context_precision_min:    0.60   # min RAGAS context precision score
  latency_p95_max_seconds:  15.0   # max 95th-percentile latency in seconds
  cost_per_query_max_usd:   0.02   # max estimated cost per query

evaluation:
  model:            gpt-4o                 # LLM for RAG answers
  judge_model:      gpt-4o-mini            # cheaper model for RAGAS evaluation judge
  embedding_model:  text-embedding-3-small # embedding model for Pinecone
  chunk_size:       1000                   # characters per chunk
  chunk_overlap:    150                    # overlap between adjacent chunks
  top_k_retrieval:  8                      # chunks retrieved per question
  pinecone_index:   asullmeval             # Pinecone cloud vector index name
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

## Live Demo

1. **Open the live dashboard:**
   ```
   https://your-vercel-url.vercel.app
   ```

2. **Make a bad change to trigger a failure:**
   ```bash
   # Open config.yaml and raise the faithfulness threshold
   # Change: faithfulness_min: 0.80 → faithfulness_min: 0.99
   git add config.yaml
   git commit -m "test: raise faithfulness threshold"
   git push
   ```

3. **Watch GitHub Actions catch the failure:**
   Go to the **Actions** tab → you will see a red ✗ FAIL

4. **Dashboard auto-updates to show FAIL**
   The CI commits the updated `latest_report.json` back to the repo.
   Reload the Vercel dashboard — it now shows the failed gate in red.

5. **Revert the change:**
   ```bash
   # Change: faithfulness_min: 0.99 → faithfulness_min: 0.80
   git add config.yaml
   git commit -m "fix: restore faithfulness threshold"
   git push
   ```
   Pipeline goes green again and dashboard updates automatically.

---

## Extending the Pipeline

### Add a new knowledge base document

1. Drop a `.txt` file into `data/knowledge_base/`
2. Clear the Pinecone index (or upsert new vectors) so it includes the new document
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
| Hallucination Rate | 0.1000 | <= 0.10 | PASS |
| Answer Relevancy | 0.9255 | >= 0.75 | PASS |
| Faithfulness | 0.9000 | >= 0.80 | PASS |
| Context Precision | 0.9333 | >= 0.60 | PASS |
| Latency P95 | 2.006 s | <= 15.0 s | PASS |
| Cost Per Query | $0.0087 | <= $0.02 | PASS |
