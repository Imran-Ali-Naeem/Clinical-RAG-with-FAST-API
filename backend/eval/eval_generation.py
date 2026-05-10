"""
Standalone generation evaluation (no API server).

Pipeline: same embedding + hybrid (+ optional rerank) as production, then Groq generator + Groq judge.
Judge scores (1–5): coherence, relevance, factual_grounding.

Requires: GROQ_API_KEY in environment or backend/.env
Optional: reranker model download on first run (same as production).

Run:
    cd backend && python eval/eval_generation.py

Output: backend/results/generation_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

import eval_common as ec

from dotenv import load_dotenv
from groq import Groq

load_dotenv(ec.BACKEND_ROOT / ".env")
load_dotenv()

TOP_K_DEFAULT = 5
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("MODEL_GROQ", "llama-3.3-70b-versatile")

TEST_CASES = [
    {
        "query": "What are the symptoms of bacterial pneumonia?",
        "expected_keywords": ["fever", "cough", "shortness of breath", "chest pain"],
    },
    {
        "query": "Describe the presentation of a patient with NSTEMI",
        "expected_keywords": ["chest pain", "troponin", "ECG", "cardiac"],
    },
    {
        "query": "What are common findings in type 2 diabetes patients?",
        "expected_keywords": ["glucose", "insulin", "polyuria", "diabetes"],
    },
    {
        "query": "Explain heart failure with reduced ejection fraction",
        "expected_keywords": ["ejection fraction", "cardiac", "dyspnea", "edema"],
    },
    {
        "query": "What neurological symptoms are seen in stroke patients?",
        "expected_keywords": ["weakness", "numbness", "speech", "hemiplegia"],
    },
]


def format_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Document {i}]\n"
            f"Disease: {doc.metadata.get('disease_category', 'unknown')}\n"
            f"Diagnosis: {doc.metadata.get('diagnosis', 'unknown')}\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n".join(parts)


def generate_answer(query: str, context: str, client: Groq) -> str:
    system = """You are a clinical decision support assistant.
Answer ONLY from the provided context.
If insufficient evidence, say so explicitly.
Flag critical findings with [CRITICAL FINDING]."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


JUDGE_PROMPT = """You are an expert clinical NLP evaluator.

Given:
- Query: {query}
- Context (excerpt): {context}
- Answer: {answer}

Score the answer on these 3 dimensions (1-5 each, integers only):

1. coherence: Is the answer well-structured and readable?
2. relevance: Does the answer directly address the query?
3. factual_grounding: Is every claim traceable to the provided context (no hallucination)?

Respond ONLY with valid JSON, no markdown or extra text:
{{"coherence": <int 1-5>, "relevance": <int 1-5>, "factual_grounding": <int 1-5>, "reasoning": "<one sentence>"}}"""


def _score_1_5(x) -> int:
    try:
        v = int(round(float(x)))
    except (TypeError, ValueError):
        return 0
    return max(1, min(5, v))


def judge_answer(query: str, context: str, answer: str, client: Groq) -> dict:
    excerpt = context[:12000] if len(context) > 12000 else context
    prompt = JUDGE_PROMPT.format(query=query, context=excerpt, answer=answer)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--no-rerank", action="store_true", help="Use hybrid only (no cross-encoder)")
    args = parser.parse_args()

    if not GROQ_KEY.strip():
        print("ERROR: GROQ_API_KEY not set. Generation eval requires Groq for generation + judge.")
        print("Set it in backend/.env or the environment, then re-run.")
        sys.exit(1)

    chunks, faiss_index, docstore, mapping, bm25, embedder, reranker = ec.load_artifacts(
        load_reranker=not args.no_rerank
    )
    client = Groq(api_key=GROQ_KEY)
    k = args.top_k

    rows = []
    retrieval_mode = "hybrid_rerank" if reranker else "hybrid"
    print(f"\nGeneration eval: {len(TEST_CASES)} cases, retrieval={retrieval_mode}, K={k}\n")

    for item in TEST_CASES:
        query = item["query"]
        print(f"Query: {query}")

        if reranker is not None:
            docs = ec.retrieve_hybrid_rerank(
                query, chunks, faiss_index, docstore, mapping, bm25, embedder, reranker, k
            ) or []
        else:
            docs = ec.retrieve_hybrid(query, chunks, faiss_index, docstore, mapping, bm25, embedder, k)

        context = format_context(docs)
        answer = generate_answer(query, context, client)
        print(f"  Answer: {answer[:120]}...")

        try:
            scores = judge_answer(query, context, answer, client)
            coherence = _score_1_5(scores.get("coherence", 0))
            relevance = _score_1_5(scores.get("relevance", 0))
            factual = _score_1_5(scores.get("factual_grounding", 0))
            reasoning = str(scores.get("reasoning", ""))
            avg_score = round((coherence + relevance + factual) / 3, 3)
        except Exception as e:
            print(f"  Judge parsing failed: {e}")
            coherence = relevance = factual = avg_score = 0
            reasoning = f"parse error: {e}"

        print(f"  Scores → coherence={coherence} relevance={relevance} factual_grounding={factual} avg={avg_score}\n")

        rows.append(
            {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "answer_excerpt": answer[:400].replace("\n", " "),
                "coherence": coherence,
                "relevance": relevance,
                "factual_grounding": factual,
                "avg_score": avg_score,
                "reasoning": reasoning,
                "docs_retrieved": len(docs),
            }
        )

    ec.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ec.RESULTS_DIR / "generation_scores.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("─" * 50)
    print("AVERAGES (1-5):")
    for metric in ["coherence", "relevance", "factual_grounding", "avg_score"]:
        avg = sum(r[metric] for r in rows) / len(rows)
        print(f"  {metric:<22} {avg:.3f}")

    print(f"\nSaved → {out_path} ✅")


if __name__ == "__main__":
    main()
