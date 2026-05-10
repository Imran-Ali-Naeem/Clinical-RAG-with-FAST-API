from langchain_groq import ChatGroq

from app.core.config import settings

_EXPANSION_SYSTEM = """You help retrieval over MIMIC-style clinical notes.
Notes often avoid textbook headings: prefer phrases that appear in charts (e.g. sepsis → infection hypotension lactate levo norepinephrine vasopressor shock SIRS bacteremia).
Output ONLY extra search keywords and short phrases. No sentences. No explanation. Max 35 words."""


def expand_retrieval_query(query: str) -> str:
    """Append ICU/note-style terms for hybrid retrieval. Falls back to the original query on error or if disabled."""
    if not settings.QUERY_EXPANSION_ENABLED or not (settings.GROQ_API_KEY or "").strip():
        return query
    try:
        llm = ChatGroq(
            model=settings.MODEL_GROQ,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.0,
        )
        msg = llm.invoke(
            [
                ("system", _EXPANSION_SYSTEM),
                ("human", f"Query: {query}\nExtra terms:"),
            ]
        )
        extra = (msg.content or "").strip()
        if not extra:
            return query
        return f"{query} {extra}"
    except Exception:
        return query
