from typing import Generator

from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import RetrievedDoc, PipelineResult
from app.services.confidence import compute_retrieval_confidence
from app.services.llm_service import route_llm, get_llm, build_chain
from app.services.query_expansion import expand_retrieval_query
from app.services.retriever import retrieve_ranked_for_rag


def format_context(docs: list[RetrievedDoc]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        part = (
            f"[Document {i}]\n"
            f"Source       : {meta.get('source', 'unknown')}\n"
            f"Disease      : {meta.get('disease_category', 'unknown')}\n"
            f"Diagnosis    : {meta.get('diagnosis', 'unknown')}\n"
            f"Retrieval    : {doc.method} (score: {doc.score})\n"
            f"Content      :\n{doc.content}\n"
        )
        parts.append(part)
    return "\n" + "─" * 60 + "\n".join(parts)


SYSTEM_PROMPT = """You are a clinical decision support assistant.
Answer using ONLY the provided clinical context. Documents are labeled [Document 1], [Document 2], … — cite these labels exactly when you reference evidence.

Rules:
1. Do not use outside or general medical knowledge; every clinical claim must be traceable to the context.
2. If the context is too thin to answer reliably, set **Summary** to exactly: Insufficient evidence in the provided clinical context. Set **Key findings**, **Evidence**, and **Analysis** to exactly: Not applicable. Set **Critical flags** to exactly: None.
3. When you answer from context, cite which document(s) support each point (e.g. Document 1, Documents 2 and 4).
4. Be concise and clinically precise; prefer short bullets where appropriate.
5. If clinical evidence in the context strongly supports a diagnosis or conclusion, state it directly. Avoid hedging when evidence is sufficient (still cite documents).

Output format — use these Markdown headings every time (no sections skipped):

**Summary:** 2–4 sentences tying the answer to what appears in context.

**Key findings:**
- Bullet points grounded in cited documents only.

**Evidence:** State which Document number(s) support the main conclusions (you may briefly name the diagnosis / label shown in context for each).

**Critical flags:** Abnormal vitals, critical labs, or emergency-related findings mentioned in context; prefix each with [CRITICAL FINDING] if present. Write exactly None if none.

**Analysis:** Add clinical reasoning that is NOT already stated in Summary or Key findings — still grounded ONLY in the documents (e.g. how findings across documents relate, spectrum or subtype implied by the excerpts, or explicit limits of what the retrieved text supports). Do not repeat prior sections. Do not introduce facts absent from context. If no additive, non-repetitive point is possible, write exactly one short sentence: Additional synthesis is limited; the excerpts are fully summarized above.
"""

HUMAN_PROMPT = """Clinical context:
{context}

Clinical query: {query}

Produce the structured response (headings exactly as instructed)."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def run_rag_pipeline(query: str, top_k: int = 5) -> PipelineResult:
    retrieval_query = expand_retrieval_query(query)
    docs = retrieve_ranked_for_rag(query, retrieval_query, top_k)
    confidence = compute_retrieval_confidence(docs)
    context = format_context(docs)
    result = route_llm(prompt, context, query)
    return PipelineResult(
        query=query,
        answer=result["answer"],
        sources=docs,
        retrieval_confidence=confidence,
        provider_used=result["provider_used"],
        fallback_triggered=result["fallback_triggered"],
        error_summary=result["error_summary"],
    )


def stream_rag_pipeline(query: str, top_k: int = 5) -> Generator:
    import json

    # step 1 — expand + wide retrieve + rerank + relative filter
    retrieval_query = expand_retrieval_query(query)
    docs = retrieve_ranked_for_rag(query, retrieval_query, top_k)
    confidence = compute_retrieval_confidence(docs)
    context = format_context(docs)

    # step 2 — try gemini first, fallback to groq
    fallback_triggered = False
    error_summary = None
    provider_used = None

    for prefer in ["gemini", "groq"]:
        try:
            llm, provider_used = get_llm(prefer=prefer)
            chain = build_chain(llm, prompt)

            # step 3 — stream tokens
            for chunk in chain.stream({"context": context, "query": query}):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            # step 4 — emit final metadata event
            metadata = {
                "type": "metadata",
                "provider_used": provider_used,
                "fallback_triggered": fallback_triggered,
                "error_summary": error_summary,
                "sources": [doc.model_dump() for doc in docs],
                "retrieval_confidence": confidence.model_dump(),
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            yield "data: [DONE]\n\n"
            return

        except Exception as e:
            if prefer == "groq":
                yield f"data: {json.dumps({'type': 'error', 'content': f'Both LLMs failed: {type(e).__name__}'})}\n\n"
                return
            fallback_triggered = True
            error_summary = f"Gemini failed: {type(e).__name__}"
            continue