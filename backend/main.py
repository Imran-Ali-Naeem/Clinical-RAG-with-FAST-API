from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.state.runtime import state
from app.core.config import settings
from app.services.retriever import load_all

app = FastAPI(title="Clinical RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    print("Server starting up...")

    # check API keys
    state.gemini_key_present = bool(settings.GOOGLE_API_KEY)
    state.groq_key_present = bool(settings.GROQ_API_KEY)

    # load indexes and model
    load_all()

    state.is_ready = True
    print("Startup complete.")

@app.get("/")
async def root():
    return {"message": "Clinical RAG API is running"}