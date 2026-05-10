import pickle
import faiss
import hashlib
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.models.schemas import RetrievedDoc
from app.state.runtime import state

DATA_DIR = "data"

# ─── Loader ───────────────────────────────────────────────────────────────────

def load_all():
    # Load chunks (for BM25 — list of LangChain Documents)
    with open(f"{DATA_DIR}/chunks.pkl", "rb") as f:
        state.chunks = pickle.load(f)
    print(f"Chunks loaded: {len(state.chunks)}")

    # Load FAISS index
    state.faiss_index = faiss.read_index(f"{DATA_DIR}/faiss_index/index.faiss")

    # Load LangChain's tuple (InMemoryDocstore, {int: uuid})
    with open(f"{DATA_DIR}/faiss_index/index.pkl", "rb") as f:
        data = pickle.load(f)
    state.faiss_docstore = data[0]   # InMemoryDocstore
    state.faiss_mapping  = data[1]   # {faiss_int_id: uuid_string}
    state.faiss_loaded = True
    print("FAISS loaded")

    # Load BM25
    with open(f"{DATA_DIR}/bm25_index.pkl", "rb") as f:
        state.bm25_index = pickle.load(f)
    state.bm25_loaded = True
    print("BM25 loaded")

    # Load embedding model
    state.embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    state.embedding_loaded = True
    print("Embedding model loaded")

    try:
        state.reranker = CrossEncoder(settings.RERANK_MODEL)
        state.reranker_loaded = True
        print(f"Reranker loaded: {settings.RERANK_MODEL}")
    except Exception as e:
        state.reranker = None
        state.reranker_loaded = False
        print(f"Reranker failed to load ({e}); continuing without reranking")


# ─── BM25 Retrieval ───────────────────────────────────────────────────────────

def retrieve_bm25(query: str, top_k: int) -> list[RetrievedDoc]:
    tokenized_query = query.lower().split()
    scores          = state.bm25_index.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = state.chunks[idx]
        results.append(RetrievedDoc(
            content=chunk.page_content,
            metadata=chunk.metadata,
            score=float(scores[idx]),
            method="bm25"
        ))
    return results


# ─── Dense Retrieval ──────────────────────────────────────────────────────────

def retrieve_dense(query: str, top_k: int) -> list[RetrievedDoc]:
    query_embedding = state.embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = state.faiss_index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        # step 1 — get uuid from int index
        uuid = state.faiss_mapping[idx]
        # step 2 — get Document from docstore using uuid
        chunk = state.faiss_docstore.search(uuid)
        results.append(RetrievedDoc(
            content=chunk.page_content,
            metadata=chunk.metadata,
            score=float(1 / (1 + dist)),
            method="dense"
        ))
    return results


# ─── RRF Fusion ───────────────────────────────────────────────────────────────


def fuse_rrf(bm25_docs: list[RetrievedDoc], dense_docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    rrf_scores: dict[str, float] = {}
    doc_map:    dict[str, RetrievedDoc] = {}
    bm25_keys:  set[str] = set()
    dense_keys: set[str] = set()

    def make_key(doc: RetrievedDoc) -> str:
        return hashlib.md5(doc.content.encode()).hexdigest()

    # score from BM25 ranks (1× weight)
    for rank, doc in enumerate(bm25_docs):
        key = make_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        doc_map[key] = doc
        bm25_keys.add(key)

    # score from dense ranks (2× weight vs BM25)
    for rank, doc in enumerate(dense_docs):
        key = make_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 2 / (60 + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc
        dense_keys.add(key)

    # sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for k in sorted_keys:
        # annotate method correctly
        in_bm25  = k in bm25_keys
        in_dense = k in dense_keys
        if in_bm25 and in_dense:
            method = "hybrid"
        elif in_bm25:
            method = "bm25"
        else:
            method = "dense"

        doc = doc_map[k]
        results.append(RetrievedDoc(
            content=doc.content,
            metadata=doc.metadata,
            score=round(rrf_scores[k], 6),
            method=method
        ))
    return results

# ─── Hybrid Retrieval ─────────────────────────────────────────────────────────

def retrieve_hybrid(query: str, top_k: int) -> list[RetrievedDoc]:
    bm25_docs  = retrieve_bm25(query, top_k)
    dense_docs = retrieve_dense(query, top_k)
    fused      = fuse_rrf(bm25_docs, dense_docs)
    return fused[:top_k]


def _relative_rerank_filter(
    pairs: list[tuple[RetrievedDoc, float]],
    spread_max: float,
) -> list[tuple[RetrievedDoc, float]]:
    """Drop chunks far below the best hit: min–max normalize rerank logits, then keep norm >= 1 - spread_max."""
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s + 1e-8
    normalized = [(s - min_s) / denom for s in scores]
    # best hit has norm 1.0; keep if within `spread_max` of best on the [0,1] scale
    kept = [
        (d, s)
        for (d, s), norm in zip(pairs, normalized)
        if (1.0 - norm) <= spread_max
    ]
    return kept if kept else list(pairs)


def retrieve_ranked_for_rag(
    user_query: str,
    retrieval_query: str,
    final_top_k: int,
) -> list[RetrievedDoc]:
    """
    Wide hybrid retrieval on `retrieval_query`, cross-encoder rerank with `user_query`,
    then relative filtering vs. the best hit in the pool. Scores returned are reranker logits.
    """
    n = max(final_top_k, settings.RETRIEVAL_CANDIDATE_K)
    docs = retrieve_hybrid(retrieval_query, n)
    if not docs:
        return []

    if not state.reranker:
        return docs[:final_top_k]

    pairs_in = [[user_query, d.content] for d in docs]
    raw_scores = state.reranker.predict(
        pairs_in,
        batch_size=16,
        show_progress_bar=False,
    )
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()

    scored = list(zip(docs, (float(s) for s in raw_scores)))
    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = _relative_rerank_filter(
        scored,
        settings.RERANK_RELATIVE_SPREAD_MAX,
    )
    # preserve rerank order
    filtered.sort(key=lambda x: x[1], reverse=True)
    out: list[RetrievedDoc] = []
    for doc, rscore in filtered[:final_top_k]:
        out.append(
            RetrievedDoc(
                content=doc.content,
                metadata=doc.metadata,
                score=rscore,
                method=f"{doc.method}→rerank",
            )
        )
    return out