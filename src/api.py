"""
PARTE 3: REST API with FastAPI

Endpoints:
- POST /ask: Answer questions
- GET /health: Health check
- GET /stats: Vector store statistics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from .rag_engine import RAGEngine
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .config import settings


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., description="The question to answer", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of context chunks to retrieve", ge=1, le=10)


class SourceInfo(BaseModel):
    """Source information"""
    chunk_id: int
    source: str
    score: float
    preview: str


class QuestionResponse(BaseModel):
    """Response model for questions"""
    answer: str
    context: List[str]
    sources: List[SourceInfo]
    confidence: float
    has_sufficient_context: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    vector_store_loaded: bool


class StatsResponse(BaseModel):
    """Statistics response"""
    num_vectors: int
    num_chunks: int
    embedding_dim: int
    is_trained: bool


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="AI Chatbot with Retrieval Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine

    try:
        print("Initializing RAG Engine...")

        # Initialize components
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()

        # Try to load existing vector store
        try:
            vector_store.load()
            print("Loaded existing vector store")
        except FileNotFoundError:
            print("No existing vector store found. Please run the ingestion pipeline first.")
            print("The API will start but /ask endpoint will not work until vector store is created.")

        # Initialize RAG engine
        rag_engine = RAGEngine(
            embedding_generator=embedding_generator,
            vector_store=vector_store
        )

        print("RAG Engine initialized successfully")

    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")
        print("API will start in degraded mode")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint

    Returns API status and vector store availability
    """
    vector_store_loaded = False

    if rag_engine and rag_engine.vector_store:
        stats = rag_engine.vector_store.get_stats()
        vector_store_loaded = stats['num_vectors'] > 0

    return HealthResponse(
        status="healthy" if vector_store_loaded else "degraded",
        message="API is running" if vector_store_loaded else "API is running but vector store is not loaded",
        vector_store_loaded=vector_store_loaded
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """
    Get vector store statistics

    Returns information about the indexed documents
    """
    if not rag_engine or not rag_engine.vector_store:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    stats = rag_engine.vector_store.get_stats()

    return StatsResponse(**stats)


@app.post("/ask", response_model=QuestionResponse, tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on the indexed documents

    The system will:
    1. Retrieve relevant context from the vector store
    2. Assess context quality
    3. Generate an answer using LLM
    4. Return the answer with sources and confidence score
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not rag_engine.vector_store or rag_engine.vector_store.get_stats()['num_vectors'] == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Please run the ingestion pipeline first."
        )

    try:
        # Update top_k if provided
        if request.top_k:
            original_top_k = settings.top_k_chunks
            settings.top_k_chunks = request.top_k

        # Get answer
        result = rag_engine.answer_question(request.question)

        # Restore original top_k
        if request.top_k:
            settings.top_k_chunks = original_top_k

        # Convert to response model
        response = QuestionResponse(
            answer=result['answer'],
            context=result['context_used'],
            sources=[SourceInfo(**source) for source in result['sources']],
            confidence=result['confidence'],
            has_sufficient_context=result['has_sufficient_context']
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


def start_api(host: str = None, port: int = None):
    """
    Start the API server

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    host = host or settings.api_host
    port = port or settings.api_port

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_api()
