from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    MODEL_GEMINI: str = "gemini-2.5-flash"
    MODEL_GROQ: str = "llama-3.3-70b-versatile"
    TOP_K_DEFAULT: int = 5

    # Retrieval → rerank → relative filter (see retriever.retrieve_ranked_for_rag)
    RETRIEVAL_CANDIDATE_K: int = 40
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # After min–max normalizing rerank scores in the pool: keep if (1 - norm) <= this (tighter = fewer chunks)
    RERANK_RELATIVE_SPREAD_MAX: float = 0.22
    QUERY_EXPANSION_ENABLED: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()