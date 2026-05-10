from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.state.runtime import state
from app.models.schemas import QueryRequest, RetrieveResponse, PipelineResult
from app.services.confidence import compute_retrieval_confidence
from app.services.pipeline import run_rag_pipeline, stream_rag_pipeline
from app.services.query_expansion import expand_retrieval_query
from app.services.retriever import retrieve_ranked_for_rag

router = APIRouter(prefix="/api", tags=["api"])


# ─── GET /api/health ──────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {
        "status": "ok" if state.is_ready else "loading",
        "components": {
            "faiss":           "loaded" if state.faiss_loaded else "not loaded",
            "bm25":            "loaded" if state.bm25_loaded else "not loaded",
            "embedding_model": "loaded" if state.embedding_loaded else "not loaded",
            "reranker":        "loaded" if state.reranker_loaded else "not loaded",
            "gemini_key":      "present" if state.gemini_key_present else "missing",
            "groq_key":        "present" if state.groq_key_present else "missing",
        }
    }


# ─── POST /api/retrieve ───────────────────────────────────────────────────────

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: QueryRequest):
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Server is still loading")
    retrieval_query = expand_retrieval_query(request.query)
    docs = retrieve_ranked_for_rag(request.query, retrieval_query, request.top_k)
    return RetrieveResponse(
        query=request.query,
        top_k=request.top_k,
        results=docs,
        retrieval_confidence=compute_retrieval_confidence(docs),
    )


# ─── POST /api/query ──────────────────────────────────────────────────────────

@router.post("/query", response_model=PipelineResult)
async def query(request: QueryRequest):
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Server is still loading")
    result = run_rag_pipeline(request.query, request.top_k)
    return result


# ─── POST /api/query/stream ───────────────────────────────────────────────────

@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Server is still loading")
    return StreamingResponse(
        stream_rag_pipeline(request.query, request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )