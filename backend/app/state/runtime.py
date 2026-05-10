class RuntimeState:
    faiss_index = None
    faiss_mapping = None
    bm25_index = None
    chunks = None
    embedder = None
    faiss_docstore = None
    reranker = None

    # health flags
    faiss_loaded = False
    bm25_loaded = False
    embedding_loaded = False
    reranker_loaded = False
    gemini_key_present = False
    groq_key_present = False
    is_ready = False


state = RuntimeState()