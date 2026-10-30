# FastAPI application for production-ready RAG deployment with Modern Rounded Frontend
import os
import logging
import time
from typing import List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
import support_rag

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_fastapi")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")


# Request and Response Schemas
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question to be answered from knowledge base"
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return cleaned


class SourceItem(BaseModel):
    source: str
    page: Optional[Any] = None


class AskResponse(BaseModel):
    status: str = Field(default="success")
    question: str
    answer: str
    sources: List[SourceItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    retriever_initialized: bool
    backend: str
    model: str = "qwen/qwen3.8-27b"


# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI RAG Service...")
    try:
        support_rag.initialize_retriever()
        logger.info("Knowledge base vector store loaded successfully.")
    except Exception as e:
        logger.warning(f"Knowledge base warm-up deferred: {e}")
    yield
    logger.info("Shutting down FastAPI RAG Service...")


# FastAPI Application Setup
app = FastAPI(
    title="Customer Support RAG API - Experiment 7 & 8",
    description="Production-ready FastAPI service wrapping RAG pipeline with GroqCloud LLM and modern frontend UI.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Frontend Mounting
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# API Endpoints
@app.get("/", tags=["Frontend"], include_in_schema=False)
def serve_frontend():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"status": "ok", "message": "FastAPI RAG API Running"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    is_ready = support_rag._retriever is not None
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        retriever_initialized=is_ready,
        backend="groqcloud",
        model=model_name
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(request: AskRequest):
    logger.info(f"Received question: {request.question[:80]}...")
    request_start = time.perf_counter()

    try:
        rag_output = support_rag.run_rag_pipeline(request.question)
        total_time = round((time.perf_counter() - request_start) * 1000, 2)
        logger.info(f"Request processed successfully in {total_time} ms.")

        return AskResponse(
            status="success",
            question=request.question,
            answer=rag_output["answer"],
            sources=[SourceItem(**s) for s in rag_output["sources"]],
            latency_ms=rag_output["latency_ms"]
        )

    except ValueError as val_err:
        logger.error(f"Configuration or input error: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )

    except Exception as err:
        logger.error(f"Internal backend failure: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer from the backend LLM service. Please check API key, quota, or network connectivity."
        )


# Server Runner
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 enables access from AWS EC2 external IP and local network
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

