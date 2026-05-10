# DiReCT — Diagnostic Reasoning for Clinical Notes
> A production-grade RAG system built on MIMIC-IV-Ext DiReCT dataset for clinical query answering and diagnostic reasoning.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-18-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Kaggle Preprocessing](#kaggle-preprocessing)
- [Running Locally](#running-locally)
- [Evaluation Results](#evaluation-results)
- [API Endpoints](#api-endpoints)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)

---

## Overview

DiReCT is a **Retrieval-Augmented Generation (RAG)** system that enables clinical query answering over physician-annotated clinical notes. It combines:

- **Hybrid retrieval** (BM25 sparse + FAISS dense) with Reciprocal Rank Fusion
- **Cross-encoder reranking** for precision
- **Query expansion** (Groq) to bridge natural-language queries and note-style vocabulary
- **Dual LLM setup** — Gemini (`MODEL_GEMINI`, primary) with Groq Llama 3.3 70B (automatic fallback)
- **Retrieval confidence** on each response (retrieval agreement signal, not diagnostic certainty)

The backend answers **only from retrieved context** (explicit prompt constraint — no outside medical knowledge). When context is insufficient, the model returns a structured “insufficient evidence” response.

---

## Architecture

```
User Query
    │
    ▼
Query Expansion (Groq, optional — QUERY_EXPANSION_ENABLED + GROQ_API_KEY)
    │  Richer retrieval query for BM25 + dense recall
    ▼
Wide Hybrid Retrieval (default 40 candidates, RETRIEVAL_CANDIDATE_K)
    ├── BM25 sparse retrieval (rank-bm25)
    └── FAISS dense retrieval (pritamdeka/S-PubMedBert-MS-MARCO)
         └── RRF Fusion (dense 2× weight vs BM25, k=60)
    │
    ▼
Cross-Encoder Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
    │  Scores (original user query, chunk) pairs
    ▼
Relative Filter (RERANK_RELATIVE_SPREAD_MAX, default 0.22)
    │  Drops chunks far below best hit (normalized spread)
    ▼
Top-K Documents (default 5) + Retrieval Confidence
    │
    ▼
LLM Generation (structured Markdown sections)
    ├── Primary:  Google Gemini (GOOGLE_API_KEY / MODEL_GEMINI)
    └── Fallback: Groq (GROQ_API_KEY / MODEL_GROQ)
    │
    ▼
Structured Clinical Response
    ├── Summary, Key findings, Evidence (Document N citations)
    ├── Critical flags
    └── Analysis
```

---

## Dataset

**MIMIC-IV-Ext DiReCT v1.0.0** — 511 physician-annotated clinical notes

| Property | Value |
|---|---|
| Total notes | 511 |
| Disease categories | 25 |
| PDD categories | 55 |
| Avg note length | 489 words |
| Format | JSON (SOAP structure) |
| Source | PhysioNet (requires credentialed access) |

**25 Disease Categories include:** Acute Coronary Syndrome, Alzheimer, Aortic Dissection, Asthma, Atrial Fibrillation, COPD, Cardiomyopathy, Diabetes, Epilepsy, Gastritis, Heart Failure, Hypertension, Migraine, Multiple Sclerosis, Pneumonia, Pulmonary Embolism, Stroke, Thyroid Disease, Tuberculosis, and more.

> ⚠️ **Note:** MIMIC-IV data requires credentialed PhysioNet access. This repository does not include raw data or patient information.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Embedding Model** | `pritamdeka/S-PubMedBert-MS-MARCO` |
| **Vector Store** | FAISS (faiss-cpu) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable via `RERANK_MODEL`) |
| **Primary LLM** | Google Gemini (`MODEL_GEMINI`, default `gemini-2.5-flash`) |
| **Fallback LLM** | Groq Llama 3.3 (`MODEL_GROQ`, default `llama-3.3-70b-versatile`) |
| **Framework** | LangChain (chains / prompts) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Vite + TypeScript (plain CSS, no Tailwind in repo) |
| **Preprocessing** | Kaggle notebook (`kaggle/direct_preprocessing.ipynb`) |

---

## Project Structure

```
Clinical-RAG/   (repo root name may vary)
├── backend/
│   ├── main.py                     # FastAPI app entry point
│   ├── requirements.txt
│   ├── .env.example                # Env template (matches config.py)
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── core/config.py          # GOOGLE_API_KEY, retrieval settings, etc.
│   │   ├── models/schemas.py       # PipelineResult, RetrievalConfidence, …
│   │   ├── services/
│   │   │   ├── pipeline.py         # RAG orchestration + prompting
│   │   │   ├── retriever.py        # Hybrid + rerank + relative filter
│   │   │   ├── llm_service.py      # Gemini + Groq fallback
│   │   │   ├── query_expansion.py  # Groq expansion for retrieval
│   │   │   └── confidence.py       # Retrieval confidence helper
│   │   └── state/runtime.py
│   ├── data/                       # Artifacts (gitignored — build via notebook)
│   ├── eval/
│   │   ├── eval_common.py
│   │   ├── eval_retrieval.py
│   │   ├── eval_generation.py
│   │   └── save_results.py
│   └── results/                    # CSV outputs from eval (often gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts           # REST + base URL
│   │   ├── api/stream.ts           # SSE query/stream
│   │   ├── types/api.ts
│   │   └── components/             # QueryInput, AnswerPanel, SourcesPanel, …
│   ├── package.json
│   └── vite.config.ts
├── kaggle/
│   └── direct_preprocessing.ipynb
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.12+
- Node.js 18+
- Google AI Studio API key for Gemini ([Google AI Studio](https://aistudio.google.com)) — stored as **`GOOGLE_API_KEY`** (see `app/core/config.py`)
- Groq API key ([console.groq.com](https://console.groq.com)) — optional for fallback LLM and required for query expansion / eval generation
- MIMIC-IV-Ext DiReCT dataset (PhysioNet credentialed access) if rebuilding artifacts

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment variables
```bash
cp .env.example .env
# Edit .env — names below match pydantic Settings in app/core/config.py
```

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key

MODEL_GEMINI=gemini-2.5-flash
MODEL_GROQ=llama-3.3-70b-versatile

RETRIEVAL_CANDIDATE_K=40
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_RELATIVE_SPREAD_MAX=0.22
QUERY_EXPANSION_ENABLED=true
```

> **Do not use `GEMINI_API_KEY`** unless you add a field alias in code — the backend expects **`GOOGLE_API_KEY`** for `ChatGoogleGenerativeAI`.

### 4. Generate artifacts (Kaggle)
See [Kaggle Preprocessing](#kaggle-preprocessing). Place outputs under `backend/data/`:

```
backend/data/
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── bm25_index.pkl
└── chunks.pkl
```

### 5. Frontend setup
```bash
cd frontend
npm install
```

The dev client defaults to **`http://127.0.0.1:8001/api`** (`frontend/src/api/client.ts`). Match your `uvicorn` port or change that constant.

---

## Kaggle Preprocessing

Heavy steps (embeddings, indexes) run on Kaggle (e.g. free T4 GPU).

### Steps
1. Upload `kaggle/direct_preprocessing.ipynb` to Kaggle
2. Add MIMIC-IV-Ext DiReCT dataset to the notebook
3. Run all cells in order (imports → load notes → SOAP/chunking → deps → **S-PubMedBert-MS-MARCO** embeddings → FAISS → BM25 → save)
4. Download `faiss_index/`, `bm25_index.pkl`, `chunks.pkl` into `backend/data/`

Notebook cell labels (chunk counts, timings) may vary slightly by run.

---

## Running Locally

### Start backend
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8001
```

Use any port; ensure **`frontend/src/api/client.ts`** and **`api/stream.ts`** use the same host/port (defaults in repo: **8001**).

### Start frontend
```bash
cd frontend
npm run dev
```

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API (example): [http://127.0.0.1:8001](http://127.0.0.1:8001)
- OpenAPI docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### Run evaluation (no API server)
```bash
cd backend
python eval/eval_retrieval.py              # hybrid + rerank if models load
python eval/eval_retrieval.py --no-rerank    # hybrid only
python eval/eval_generation.py             # requires GROQ_API_KEY
python eval/save_results.py                  # print CSV summaries
```

---

## Evaluation Results

Values below are from **one representative local run** (`backend/results/*.csv`); your numbers will differ after reruns or data changes.

### Retrieval Metrics (P@5, R@5, F1@5)

| Method | Precision@5 | Recall@5 | F1@5 |
|---|---|---|---|
| BM25 | 0.400 | 0.021 | 0.039 |
| Dense (FAISS) | 0.640 | 0.036 | 0.066 |
| **Hybrid (BM25+FAISS)** | **0.640** | **0.036** | **0.066** |
| Hybrid + Rerank | 0.600 | 0.032 | 0.060 |

> Eval recall uses **hits in top‑K divided by total corpus chunks** in the gold disease category, so recall is often small when categories are large and K is small. See `eval/eval_common.py`.

### Generation Metrics (1–5 scale, Groq judge)

| Metric | Score |
|---|---|
| Coherence | 5.0 / 5.0 |
| Relevance | 5.0 / 5.0 |
| Factual Grounding | 4.8 / 5.0 |
| **Overall Average** | **4.93 / 5.0** |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Status: FAISS, BM25, embeddings, reranker, API keys |
| `POST` | `/api/query` | Full RAG — `answer`, `sources`, `retrieval_confidence`, `provider_used`, … |
| `POST` | `/api/query/stream` | SSE tokens + final metadata (sources + confidence) |
| `POST` | `/api/retrieve` | Retrieval only — `results` + `retrieval_confidence` |

### Example request
```bash
curl -X POST http://127.0.0.1:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "patient with chest pain and elevated troponin", "top_k": 5}'
```

### Example response shape (`PipelineResult`)
```json
{
  "query": "patient with chest pain and elevated troponin",
  "answer": "**Summary:** ...\n\n**Key findings:**\n- ...",
  "sources": [
    {
      "content": "...",
      "metadata": { "source": "...", "disease_category": "...", "diagnosis": "..." },
      "score": 4.12,
      "method": "hybrid→rerank"
    }
  ],
  "retrieval_confidence": {
    "score": 0.7234,
    "label": "high",
    "basis": "rerank",
    "top1_top2_gap": 0.142,
    "mean_normalized_strength": 0.88,
    "num_sources": 5,
    "detail": "Retrieval confidence high (72%) from cross-encoder rerank: ..."
  },
  "provider_used": "gemini",
  "fallback_triggered": false,
  "error_summary": null
}
```

---

## Ethical Considerations

- **No patient data in repository** — Raw notes stay under PhysioNet / local processing only
- **PhysioNet compliance** — Credentialed access and DUA required for MIMIC-derived data
- **Research / demo only** — Not validated for production clinical decision support
- **Grounding policy** — Answers are constrained to retrieved excerpts; “insufficient evidence” when context is thin (see system prompt in `app/services/pipeline.py`)
- **Retrieval confidence** — Reflects retrieval score agreement, **not** diagnostic probability

---

## Future Work

- Medical-specific reranker trained on clinical data
- Support for larger MIMIC-IV subsets
- Fine-tuned embedding model on MIMIC vocabulary
- Multi-turn conversational queries
- FHIR integration for real EHR systems
- Confidence calibration with human expert validation

---

## Citation

If you use this project, please cite the original DiReCT dataset:

```
MIMIC-IV-Ext DiReCT v1.0.0
PhysioNet. https://physionet.org/content/mimic-iv-ext-direct/1.0.0/
```

---

## License

MIT License — add a `LICENSE` file at the repo root when distributing.

---

<div align="center">
Built with ❤️ using MIMIC-IV · LangChain · FastAPI · React
</div>
