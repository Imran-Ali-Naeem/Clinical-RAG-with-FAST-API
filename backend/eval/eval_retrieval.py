"""
Standalone retrieval evaluation (no API server).

Metrics: Precision@K, Recall@K, F1@K with disease_category match.
Recall@K = (# relevant in top-K) / (total chunks in corpus with that disease_category).

Run from repository root or backend/:
    cd backend && python eval/eval_retrieval.py
    python eval/eval_retrieval.py  # if cwd is backend

Outputs: backend/results/retrieval_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

import eval_common as ec

TOP_K_DEFAULT = 5

TEST_QUERIES = [
    {"query": "chest pain elevated troponin myocardial infarction", "relevant_category": "Acute Coronary Syndrome"},
    {"query": "polyuria polydipsia type 2 diabetes insulin", "relevant_category": "Diabetes"},
    {"query": "shortness of breath heart failure ejection fraction", "relevant_category": "Heart Failure"},
    {"query": "seizure epilepsy anticonvulsant medication", "relevant_category": "Epilepsy"},
    {"query": "fever cough pneumonia bacterial infection", "relevant_category": "Pneumonia"},
    {"query": "stroke hemiplegia cerebral infarction", "relevant_category": "Stroke"},
    {"query": "wheezing bronchospasm COPD exacerbation", "relevant_category": "COPD"},
    {"query": "headache migraine aura photophobia", "relevant_category": "Migraine"},
    {"query": "hypertension blood pressure antihypertensive", "relevant_category": "Hypertension"},
    {"query": "multiple sclerosis demyelination MRI lesions", "relevant_category": "Multiple Sclerosis"},
]


def main():
    parser = argparse.ArgumentParser(description="Retrieval P@K / R@K / F1@K eval")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help="K for metrics (default 5)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder hybrid_rerank column")
    args = parser.parse_args()
    k = args.top_k

    chunks, faiss_index, docstore, mapping, bm25, embedder, reranker = ec.load_artifacts(
        load_reranker=not args.no_rerank
    )
    cat_counts = ec.corpus_relevant_counts(chunks)

    rows = []
    print(f"\nRetrieval eval: {len(TEST_QUERIES)} queries, K={k}, embedder+RRF match production.\n")

    for item in TEST_QUERIES:
        query = item["query"]
        category = item["relevant_category"]
        gold = ec.normalize_category(category)
        r_corpus = cat_counts.get(gold, 0)
        if r_corpus == 0:
            print(f"⚠ No corpus chunks for category '{category}' — recall@K will be 0.")

        bm25_docs = ec.retrieve_bm25(query, chunks, bm25, k)
        dense_docs = ec.retrieve_dense(query, faiss_index, docstore, mapping, embedder, k)
        hybrid_docs = ec.retrieve_hybrid(query, chunks, faiss_index, docstore, mapping, bm25, embedder, k)

        p_b, r_b, f_b = ec.precision_recall_f1_at_k(bm25_docs, category, r_corpus, k)
        p_d, r_d, f_d = ec.precision_recall_f1_at_k(dense_docs, category, r_corpus, k)
        p_h, r_h, f_h = ec.precision_recall_f1_at_k(hybrid_docs, category, r_corpus, k)

        row = {
            "query": query,
            "relevant_category": category,
            "corpus_relevant_count": r_corpus,
            "k": k,
            "bm25_precision": p_b,
            "bm25_recall": r_b,
            "bm25_f1": f_b,
            "dense_precision": p_d,
            "dense_recall": r_d,
            "dense_f1": f_d,
            "hybrid_precision": p_h,
            "hybrid_recall": r_h,
            "hybrid_f1": f_h,
        }

        if reranker is not None:
            rr_docs = ec.retrieve_hybrid_rerank(
                query, chunks, faiss_index, docstore, mapping, bm25, embedder, reranker, k
            )
            p_rr, r_rr, f_rr = ec.precision_recall_f1_at_k(rr_docs or [], category, r_corpus, k)
            row["hybrid_rerank_precision"] = p_rr
            row["hybrid_rerank_recall"] = r_rr
            row["hybrid_rerank_f1"] = f_rr
            print(f"Query : {query[:60]}...")
            print(f"  BM25          P@{k}={p_b:.3f}  R@{k}={r_b:.3f}  F1={f_b:.3f}")
            print(f"  Dense         P@{k}={p_d:.3f}  R@{k}={r_d:.3f}  F1={f_d:.3f}")
            print(f"  Hybrid        P@{k}={p_h:.3f}  R@{k}={r_h:.3f}  F1={f_h:.3f}")
            print(f"  Hybrid+rerank P@{k}={p_rr:.3f}  R@{k}={r_rr:.3f}  F1={f_rr:.3f}\n")
        else:
            row["hybrid_rerank_precision"] = ""
            row["hybrid_rerank_recall"] = ""
            row["hybrid_rerank_f1"] = ""
            print(f"Query : {query[:60]}...")
            print(f"  BM25   P@{k}={p_b:.3f}  R@{k}={r_b:.3f}  F1={f_b:.3f}")
            print(f"  Dense  P@{k}={p_d:.3f}  R@{k}={r_d:.3f}  F1={f_d:.3f}")
            print(f"  Hybrid P@{k}={p_h:.3f}  R@{k}={r_h:.3f}  F1={f_h:.3f}\n")

        rows.append(row)

    ec.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ec.RESULTS_DIR / "retrieval_metrics.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("─" * 50)
    print("MACRO AVERAGES:")
    for prefix in ["bm25", "dense", "hybrid"]:
        avg_p = sum(r[f"{prefix}_precision"] for r in rows) / len(rows)
        avg_r = sum(r[f"{prefix}_recall"] for r in rows) / len(rows)
        avg_f = sum(r[f"{prefix}_f1"] for r in rows) / len(rows)
        print(f"  {prefix.upper():<14} P@{k}={avg_p:.3f}  R@{k}={avg_r:.3f}  F1={avg_f:.3f}")

    if reranker is not None:
        avg_p = sum(float(r["hybrid_rerank_precision"]) for r in rows) / len(rows)
        avg_r = sum(float(r["hybrid_rerank_recall"]) for r in rows) / len(rows)
        avg_f = sum(float(r["hybrid_rerank_f1"]) for r in rows) / len(rows)
        print(f"  HYBRID_RERANK  P@{k}={avg_p:.3f}  R@{k}={avg_r:.3f}  F1={avg_f:.3f}")

    print(f"\nSaved → {out_path} ✅")


if __name__ == "__main__":
    main()
