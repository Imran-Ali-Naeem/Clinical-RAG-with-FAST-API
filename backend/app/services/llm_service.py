from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.core.config import settings


def get_llm(prefer: str = "gemini"):
    if prefer == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Gemini API key is missing")
        return ChatGoogleGenerativeAI(
            model=settings.MODEL_GEMINI,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        ), "gemini"
    
    if not settings.GROQ_API_KEY:
        raise RuntimeError("Groq API key is missing")
    return ChatGroq(
        model=settings.MODEL_GROQ,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.1,
    ), "groq"

def build_chain(llm, prompt):
    return (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )


def route_llm(prompt, context: str, query: str) -> dict:
    """
    Primary router — tries Gemini first, falls back to Groq.
    Returns dict with answer, provider_used, fallback_triggered, error_summary.
    """
    fallback_triggered = False
    error_summary = None

    for prefer in ["gemini", "groq"]:
        try:
            llm, provider = get_llm(prefer=prefer)
            chain = build_chain(llm, prompt)
            answer = chain.invoke({"context": context, "query": query})
            return {
                "answer": answer,
                "provider_used": provider,
                "fallback_triggered": fallback_triggered,
                "error_summary": error_summary,
            }
        except Exception as e:
            if prefer == "groq":
                raise RuntimeError(f"Both LLMs failed: {e}")
            fallback_triggered = True
            error_summary = f"Gemini failed: {type(e).__name__}"
            continue