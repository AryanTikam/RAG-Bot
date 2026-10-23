# FastAPI application for production-ready RAG deployment
import logging
import time
from typing import List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import support_rag

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_fastapi")


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
    title="Customer Support RAG API - Experiment 7",
    description="Production-ready FastAPI service wrapping Experiment 5 RAG pipeline with GroqCloud LLM.",
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


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    is_ready = support_rag._retriever is not None
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        retriever_initialized=is_ready,
        backend="groqcloud"
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
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
