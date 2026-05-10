from typing import Any, Literal

from pydantic import BaseModel


class RetrievedDoc(BaseModel):
    content: str
    metadata: dict[str, Any]
    score: float
    method: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievalConfidence(BaseModel):
    """How strongly retrieved sources agree with the query (retrieval signal only — not diagnostic certainty)."""
    score: float
    label: Literal["low", "medium", "high"]
    basis: Literal["rerank", "rrf_fallback", "none"]
    top1_top2_gap: float | None
    mean_normalized_strength: float
    num_sources: int
    detail: str


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedDoc]
    retrieval_confidence: RetrievalConfidence


class PipelineResult(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedDoc]
    retrieval_confidence: RetrievalConfidence
    provider_used: str
    fallback_triggered: bool = False
    error_summary: str | None = None

class StreamMetadata(BaseModel):
    provider_used: str
    fallback_triggered: bool
    error_summary: str | None
    sources: list[RetrievedDoc]
    retrieval_confidence: RetrievalConfidence