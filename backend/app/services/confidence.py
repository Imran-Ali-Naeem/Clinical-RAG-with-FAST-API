"""Retrieval-only confidence from ranked source scores (not clinical certainty)."""

from typing import Literal

from app.models.schemas import RetrievedDoc, RetrievalConfidence


def compute_retrieval_confidence(docs: list[RetrievedDoc]) -> RetrievalConfidence:
    """
    Heuristic 0–1 score from the final retrieved set: rerank logits or hybrid RRF scores.
    Uses min–max normalization within the returned bundle, mean strength, top1–top2 gap, and count.
    """
    if not docs:
        return RetrievalConfidence(
            score=0.0,
            label="low",
            basis="none",
            top1_top2_gap=None,
            mean_normalized_strength=0.0,
            num_sources=0,
            detail="No sources retrieved; retrieval confidence is not applicable.",
        )

    rerank = any("→rerank" in (d.method or "") for d in docs)
    basis: Literal["rerank", "rrf_fallback"] = "rerank" if rerank else "rrf_fallback"

    scores = [float(d.score) for d in docs]
    min_s, max_s = min(scores), max(scores)
    denom = max_s - min_s + 1e-8
    norms = [(s - min_s) / denom for s in scores]
    mean_norm = sum(norms) / len(norms)

    if len(norms) >= 2:
        gap = norms[0] - norms[1]
        top1_top2_gap = float(gap)
        sep_term = min(1.0, gap * 1.5)
    else:
        top1_top2_gap = None
        sep_term = 0.72

    n_boost = min(1.0, len(docs) / 5.0)
    raw = 0.52 * mean_norm + 0.33 * sep_term + 0.15 * n_boost
    score = max(0.0, min(1.0, raw))

    if score < 0.38:
        label = "low"
    elif score < 0.65:
        label = "medium"
    else:
        label = "high"

    basis_phrase = "cross-encoder rerank" if basis == "rerank" else "hybrid rank scores"
    detail = (
        f"Retrieval confidence {label} ({score:.0%}) from {basis_phrase}: "
        f"{len(docs)} source(s), mean relative strength {mean_norm:.2f}."
    )
    if top1_top2_gap is not None:
        detail += f" Top-1 vs top-2 separation {top1_top2_gap:.2f} (normalized)."

    return RetrievalConfidence(
        score=round(score, 4),
        label=label,
        basis=basis,
        top1_top2_gap=round(top1_top2_gap, 4) if top1_top2_gap is not None else None,
        mean_normalized_strength=round(mean_norm, 4),
        num_sources=len(docs),
        detail=detail,
    )
