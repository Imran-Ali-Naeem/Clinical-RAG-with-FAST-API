"""
Shared loaders and retrieval for standalone eval scripts (no FastAPI).

Paths resolve to backend/: `data/`, `results/`.

Requirements:
  - Indexed artifacts under backend/data/ (chunks.pkl, faiss_index/, bm25_index.pkl).
  - Embedding model `pritamdeka/S-PubMedBert-MS-MARCO` in HF cache or network on first load.
  - Optional rerank: `cross-encoder/ms-marco-MiniLM-L-6-v2` (override with env RERANK_MODEL).
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Must match indexing in the pipeline that built data/faiss_index
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "40"))
RERANK_RELATIVE_SPREAD_MAX = float(os.getenv("RERANK_RELATIVE_SPREAD_MAX", "0.22"))
RRF_K = 60
BM25_RRF_WEIGHT = 1.0
DENSE_RRF_WEIGHT = 2.0

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"
RESULTS_DIR = BACKEND_ROOT / "results"


def normalize_category(name: str) -> str:
    return (name or "").strip().lower()


def corpus_relevant_counts(chunks: list) -> Counter:
    c: Counter = Counter()
    for doc in chunks:
        cat = normalize_category(doc.metadata.get("disease_category", ""))
        if cat:
            c[cat] += 1
    return c


def load_artifacts(load_reranker: bool = False):
    """Load chunks, FAISS, BM25, embedder; optionally cross-encoder for reranking."""
    print("Loading artifacts...")
    with open(DATA_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    faiss_index = faiss.read_index(str(DATA_DIR / "faiss_index" / "index.faiss"))
    with open(DATA_DIR / "faiss_index" / "index.pkl", "rb") as f:
        data = pickle.load(f)
    docstore, mapping = data[0], data[1]

    with open(DATA_DIR / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = None
    if load_reranker:
        try:
            reranker = CrossEncoder(RERANK_MODEL)
            print(f"Reranker loaded: {RERANK_MODEL}")
        except Exception as e:
            print(f"Reranker not available ({e}); hybrid_rerank metrics will be skipped.")

    print(f"Loaded {len(chunks)} chunks (embedder={EMBEDDING_MODEL}).")
    return chunks, faiss_index, docstore, mapping, bm25, embedder, reranker


def _doc_key(doc) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def retrieve_bm25(query: str, chunks, bm25, top_k: int):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def retrieve_dense(query: str, faiss_index, docstore, mapping, embedder, top_k: int):
    emb = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    _, indices = faiss_index.search(emb, top_k)
    out = []
    for idx in indices[0]:
        if idx == -1:
            continue
        out.append(docstore.search(mapping[idx]))
    return out


def fuse_rrf(bm25_docs: list, dense_docs: list) -> list:
    """Same weighting as app.services.retriever.fuse_rrf (dense 2× BM25)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}
    bm25_keys: set[str] = set()
    dense_keys: set[str] = set()

    for rank, doc in enumerate(bm25_docs):
        k = _doc_key(doc)
        scores[k] = scores.get(k, 0) + BM25_RRF_WEIGHT / (RRF_K + rank + 1)
        doc_map[k] = doc
        bm25_keys.add(k)

    for rank, doc in enumerate(dense_docs):
        k = _doc_key(doc)
        scores[k] = scores.get(k, 0) + DENSE_RRF_WEIGHT / (RRF_K + rank + 1)
        if k not in doc_map:
            doc_map[k] = doc
        dense_keys.add(k)

    sorted_keys = sorted(scores, key=lambda kv: scores[kv], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def retrieve_hybrid(query: str, chunks, faiss_index, docstore, mapping, bm25, embedder, top_k: int) -> list:
    bm25_docs = retrieve_bm25(query, chunks, bm25, top_k)
    dense_docs = retrieve_dense(query, faiss_index, docstore, mapping, embedder, top_k)
    fused = fuse_rrf(bm25_docs, dense_docs)
    return fused[:top_k]


def _relative_rerank_filter(pairs: list[tuple[object, float]], spread_max: float) -> list[tuple[object, float]]:
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    min_s, max_s = min(scores), max(scores)
    denom = max_s - min_s + 1e-8
    normalized = [(s - min_s) / denom for s in scores]
    kept = [(d, s) for (d, s), norm in zip(pairs, normalized) if (1.0 - norm) <= spread_max]
    return kept if kept else list(pairs)


def retrieve_hybrid_rerank(
    query: str,
    chunks,
    faiss_index,
    docstore,
    mapping,
    bm25,
    embedder,
    reranker: CrossEncoder,
    final_k: int,
) -> list | None:
    """Wide hybrid pool + cross-encoder rerank + relative filter (production-style)."""
    n = max(final_k, RETRIEVAL_CANDIDATE_K)
    bm25_docs = retrieve_bm25(query, chunks, bm25, n)
    dense_docs = retrieve_dense(query, faiss_index, docstore, mapping, embedder, n)
    fused = fuse_rrf(bm25_docs, dense_docs)
    candidates = fused[:n]
    if not candidates:
        return []

    pairs_in = [[query, d.page_content] for d in candidates]
    raw = reranker.predict(pairs_in, batch_size=16, show_progress_bar=False)
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    scored = list(zip(candidates, (float(s) for s in raw)))
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = _relative_rerank_filter(scored, RERANK_RELATIVE_SPREAD_MAX)
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in filtered[:final_k]]


def precision_recall_f1_at_k(
    retrieved_docs: list,
    gold_category: str,
    total_relevant_in_corpus: int,
    k: int,
) -> tuple[float, float, float]:
    """
    Disease-category match: chunk is relevant if metadata disease_category equals gold (case-insensitive).
    Precision@K = (# relevant in top-K) / K
    Recall@K = (# relevant in top-K) / (total relevant in corpus)
    """
    gold = normalize_category(gold_category)
    top = retrieved_docs[:k]
    tp = sum(
        1
        for d in top
        if normalize_category(d.metadata.get("disease_category", "")) == gold
    )
    precision = tp / k if k > 0 else 0.0
    if total_relevant_in_corpus <= 0:
        recall = 0.0
    else:
        recall = min(tp / total_relevant_in_corpus, 1.0)
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)
