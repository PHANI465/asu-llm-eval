# ASU LLM Evaluation Pipeline

An automated CI/CD pipeline that evaluates the quality of an ASU university RAG chatbot on every GitHub push. Uses **RAGAS** to score faithfulness, relevancy, and precision; enforces thresholds through quality gates; and visualises results in a live React dashboard deployed on Vercel.

> **Live Dashboard:** https://asu-llm-eval.vercel.app/
> **GitHub Repo:** https://github.com/PHANI465/asu-llm-eval

---

## Architecture

```
GitHub Push
     |
     v
GitHub Actions (eval.yml)
     |
     +---> pytest  (offline unit tests — free, fail fast)
     |
     +---> src/run_eval.py  (master orchestrator)
               |
               +---> preflight question
               |         stops with ERROR if keys / quota are broken
               |
               +---> src/rag_pipeline.py
               |         Pinecone (one namespace per knowledge-base version)
               |         + GPT-4o answers
               |
               +---> src/evaluator.py
               |         RAGAS scoring, judge calls run concurrently
               |         GPT-4o-mini judge
               |
               +---> src/quality_gates.py
               |         PASS / FAIL gates
               |
               +---> results/latest_report.json + eval_history.json
               |     (committed back to main)
               |
               v
          PASS → exit 0     FAIL / ERROR → exit 1
               |
               v
     React Dashboard on Vercel
     (auto-refreshes from GitHub)
```

### Run statuses

| Status | Meaning |
|---|---|
| **PASS** | Every quality gate was met |
| **FAIL** | At least one gate was missed — a real quality or cost regression |
| **ERROR** | The pipeline could not run (bad key, exhausted OpenAI quota, network). Says nothing about answer quality; the report, CI summary, and dashboards show the error and a hint |

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
| Error Rate | <= 0.10 | lower is better |

- **Hallucination rate** is the share of answers whose faithfulness is below `hallucination_faithfulness_threshold` (0.5). It catches individual fabricated answers that a high *average* faithfulness can hide.
- **Latency** and **cost** are measured over successful calls only, and cost uses separate input/output token prices from `config.yaml`. The RAGAS judge cost is reported separately and is not part of the gate.
- **Error rate** is the share of questions whose RAG call failed outright.
- A gate with no measurable value (e.g. nothing could be scored) **fails** — it never passes by default.

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
| Run history | `eval_history.json` (committed) + SQLite archive (local) |
| Tests | pytest |
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
│   ├── eval_history.json         # Last 50 runs (committed by CI, read by both dashboards)
│   ├── eval_history.db           # SQLite archive of every local run (gitignored)
│   └── latest_report.json        # Full report of the latest run (committed by CI)
├── src/
│   ├── project_config.py         # Shared paths, config.yaml, API keys, pricing
│   ├── rag_pipeline.py           # RAG core: load, chunk, embed, answer
│   ├── evaluator.py              # RAGAS scoring engine
│   ├── quality_gates.py          # Metrics + threshold checks + PASS/FAIL
│   ├── run_eval.py               # Master orchestrator (CI entry point)
│   └── report_summary.py         # CI log + GitHub step summary
├── tests/                        # Offline unit tests (no API keys needed)
├── config.yaml                   # Thresholds, evaluation settings, model prices
├── pytest.ini
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
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API keys

Create `.env` in the project root:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
```

Environment variables take precedence over `.env`.

### 3. Run the unit tests (free, no keys needed)

```bash
pytest
```

### 4. Run the evaluation (test mode — 10 questions)

```bash
python src/run_eval.py
```

The 10 questions are sampled evenly across all six categories (the same set every run, so trends stay comparable). Output ends with:

```
============================================
QUALITY GATE REPORT
============================================
  hallucination_rate  : PASS  (0.0000 <= 0.1)
  answer_relevancy    : PASS  (0.9407 >= 0.75)
  faithfulness        : PASS  (1.0000 >= 0.8)
  context_precision   : PASS  (0.9333 >= 0.6)
  latency_p95         : PASS  (1.384s <= 15.0s)
  cost_per_query      : PASS  ($0.0047 <= $0.02)
  error_rate          : PASS  (0.0% <= 10%)
============================================
OVERALL RESULT: PASS
All 7 gates passed. Safe to deploy.
```

### 5. Other run options

```bash
python src/run_eval.py --full                     # all 100 questions (or EVAL_FULL_RUN=true)
python src/run_eval.py --limit 3                  # quick, cheap smoke test
python src/run_eval.py --results-dir /tmp/eval    # keep experiments out of results/
```

### 6. View the local Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### 7. Run the React dashboard locally

```bash
cd dashboard-react
npm install
npm run dev
```

It reads the reports from GitHub by default. Point it elsewhere (a fork, or a local file server) with `VITE_REPORT_URL` and `VITE_HISTORY_URL`.

---

## CI/CD Setup (GitHub Actions)

### 1. Add the secrets

In your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | `sk-...` |
| `PINECONE_API_KEY` | `pcsk-...` |

The OpenAI account behind the key needs available credit — if it runs out, runs report **ERROR** with an "out of credit" hint.

### 2. Push to trigger

```bash
git add .
git commit -m "feat: add new knowledge base doc"
git push origin main
```

The workflow in `.github/workflows/eval.yml` runs on every push to `main`/`master` and on pull requests:

1. Installs dependencies and runs the unit tests
2. Runs the evaluation
3. Uploads `latest_report.json` as an artifact and writes a gate table to the run's **Summary** page
4. On `main`/`master` only, commits `results/latest_report.json` and `results/eval_history.json` so the Vercel dashboard updates. Pull request runs never overwrite the dashboard.

### 3. Manual full run

Go to **Actions → LLM Evaluation Pipeline → Run workflow** and select `full_run = true` to run all 100 questions on demand.

### 4. Download the report

After any run, go to **Actions → your run → Artifacts** and download `evaluation-report-<run_number>` to inspect `latest_report.json`.

---

## Configuration Reference

All settings live in `config.yaml`:

```yaml
quality_gates:
  hallucination_rate_max:   0.10   # max share of answers flagged as hallucinated
  answer_relevancy_min:     0.75   # min RAGAS answer relevancy score
  faithfulness_min:         0.80   # min RAGAS faithfulness score
  context_precision_min:    0.60   # min RAGAS context precision score
  latency_p95_max_seconds:  15.0   # max 95th-percentile latency in seconds
  cost_per_query_max_usd:   0.02   # max answer-model cost per query
  error_rate_max:           0.10   # max share of failed RAG calls

evaluation:
  model:            gpt-4o                 # LLM for RAG answers
  judge_model:      gpt-4o-mini            # cheaper model for RAGAS evaluation judge
  embedding_model:  text-embedding-3-small # embedding model for Pinecone
  chunk_size:       1000                   # characters per chunk
  chunk_overlap:    150                    # overlap between adjacent chunks
  top_k_retrieval:  8                      # chunks retrieved per question
  pinecone_index:   asullmeval             # Pinecone cloud vector index name
  hallucination_faithfulness_threshold: 0.5  # answers below this count as hallucinated
  test_mode_questions: 10                  # questions in a default (non --full) run
  judge_max_workers: 8                     # concurrent judge requests

pricing:                                   # USD per 1M tokens
  gpt-4o:                 {input: 2.50, output: 10.00}
  gpt-4o-mini:            {input: 0.15, output: 0.60}
  text-embedding-3-small: {input: 0.02, output: 0.00}
```

Check the prices against https://openai.com/api/pricing whenever you change models — a unit test fails if a configured model has no price.

---

## Module API Reference

### `src/rag_pipeline.py`

```python
get_answer(question: str) -> dict
# Never raises. Returns:
# {
#   "question":         str,
#   "answer":           str,          # "" when the call failed
#   "retrieved_chunks": list[str],
#   "source_documents": list[str],
#   "latency_seconds":  float,
#   "token_usage":      int,
#   "input_tokens":     int,
#   "output_tokens":    int,
#   "cost_usd":         float | None,
#   "error":            str | None,   # e.g. "RateLimitError: ..." when the call failed
# }
```

### `src/evaluator.py`

```python
evaluate_batch(results: list, references: list = None) -> tuple[list, dict]
# (scored results, judge usage {"input_tokens", "output_tokens", "cost_usd"})
# Each scored dict has: faithfulness, answer_relevancy, context_precision, evaluation_error
# Results with an "error" are not sent to the judge.
evaluate_single(result: dict, reference: str = "") -> dict
```

### `src/quality_gates.py`

```python
compute_hallucination_rate(results: list, threshold: float = None) -> float | None
compute_aggregate_metrics(results: list) -> dict
check_gates(metrics: dict, thresholds: dict = None) -> dict   # {"overall": "PASS"|"FAIL", "gates": {...}, ...}
print_gate_report(gate_results: dict) -> None
```

---

## Live Demo

1. **Open the live dashboard:**
   ```
   https://asu-llm-eval.vercel.app/
   ```

2. **Make a bad change to trigger a failure:**
   ```bash
   # Open config.yaml and raise the faithfulness threshold
   # Change: faithfulness_min: 0.80 → faithfulness_min: 1.01
   git add config.yaml
   git commit -m "test: raise faithfulness threshold"
   git push
   ```

3. **Watch GitHub Actions catch the failure:**
   Go to the **Actions** tab → you will see a red ✗ FAIL, with the gate table on the run's Summary page

4. **Dashboard auto-updates to show FAIL**
   The CI commits the updated `latest_report.json` back to the repo.
   Reload the Vercel dashboard — it now shows the failed gate in red.

5. **Revert the change:**
   ```bash
   # Change: faithfulness_min: 1.01 → faithfulness_min: 0.80
   git add config.yaml
   git commit -m "fix: restore faithfulness threshold"
   git push
   ```
   Pipeline goes green again and dashboard updates automatically.

---

## Extending the Pipeline

### Add or edit a knowledge base document

1. Drop or edit a `.txt` file in `data/knowledge_base/`
2. Add relevant Q&A pairs to `data/golden_dataset.json` (a unit test checks every entry cites an existing document)
3. Push to `main`

No manual Pinecone work is needed: each knowledge-base version is stored in its own namespace (named after a hash of the chunks and embedding model), so the next run embeds the new version automatically. Old namespaces are listed in the run log and can be deleted from the Pinecone console.

### Add a new quality gate

1. Add the threshold to `config.yaml` under `quality_gates`
2. Add a gate definition tuple to `GATE_DEFINITIONS` in `src/quality_gates.py`
3. Compute the metric in `compute_aggregate_metrics()`
4. Add a label/format entry in both dashboards (`METRIC_CONFIG` in `App.jsx`, `kpi_defs` in `dashboard/app.py`)

### Adjust thresholds

Edit `config.yaml` — no code changes required. The gate engine reads thresholds at runtime, and both dashboards draw threshold lines from the report.

---

## Sample Evaluation Results

Test-mode run (10 questions across all categories):

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Hallucination Rate | 0.0% | <= 10% | PASS |
| Answer Relevancy | 0.9407 | >= 0.75 | PASS |
| Faithfulness | 1.0000 | >= 0.80 | PASS |
| Context Precision | 0.9333 | >= 0.60 | PASS |
| Latency P95 | 1.38 s | <= 15.0 s | PASS |
| Cost Per Query | $0.0047 | <= $0.02 | PASS |
| Error Rate | 0.0% | <= 10% | PASS |

Total run cost: $0.070 (answers $0.047 + judge $0.023), 37 seconds.
