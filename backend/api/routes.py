"""
routes.py — All API endpoints for the RAG system.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import time

from backend.ingestion.pipeline import IngestionPipeline
from backend.llm import generate_answer
from backend.agent.graph import get_graph
from backend.agent.memory import memory
from backend.evaluation.logger import query_logger

router = APIRouter()

_pipeline = IngestionPipeline()


# ── Pydantic Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question", min_length=3)
    session_id: Optional[str] = Field(None, description="For multi-turn conversations")
    use_web_search: bool = Field(False, description="Allow agent to search the web")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG and how does it work?",
                "session_id": "user-123",
                "use_web_search": False
            }
        }


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    session_id: str
    latency_ms: float
    retrieval_scores: dict


class IngestRequest(BaseModel):
    url: str = Field(..., description="URL or YouTube link to ingest")
    source_name: Optional[str] = None


class IngestResponse(BaseModel):
    chunks_created: int
    source: str
    message: str


class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    avg_faithfulness: float
    avg_relevancy: float
    top_sources: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main RAG query endpoint — runs the full agentic pipeline."""
    start = time.time()
    logger.info(f"Query received: {request.question[:60]}...")

    try:
        session_id = request.session_id or f"session-{int(time.time())}"

        graph = get_graph()
        initial_state = {
            "question": request.question,
            "session_id": session_id,
            "use_web_search": request.use_web_search,
            "route": "",
            "rewritten_query": "",
            "documents": [],
            "answer": "",
            "rewrite_count": 0,
            "generation_count": 0,
            "needs_rewrite": False,
            "is_grounded": True,
        }

        final_state = graph.invoke(initial_state)

        answer = final_state.get("answer", "No answer generated.")
        docs = final_state.get("documents", [])
        latency_ms = (time.time() - start) * 1000

        memory.add_turn(session_id, request.question, answer)

        query_logger.log_query(
            question=request.question,
            answer=answer,
            route=final_state.get("route", "unknown"),
            latency_ms=latency_ms,
            num_docs=len(docs),
            rewrite_count=final_state.get("rewrite_count", 0),
        )

        sources = [
            SourceDocument(
                content=doc.page_content[:300],
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page"),
                score=doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0.0))
            )
            for doc in docs
        ]

        logger.info(f"Query answered in {latency_ms:.0f}ms")

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            retrieval_scores={}
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestRequest):
    """Ingest a URL or YouTube video into the vector store."""
    logger.info(f"Ingest URL request: {request.url}")
    result = _pipeline.ingest_url(request.url)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return IngestResponse(
        chunks_created=result["chunks_created"],
        source=result["source"],
        message=f"Ingested {result['docs_loaded']} pages -> {result['chunks_created']} chunks stored."
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF or CSV file."""
    logger.info(f"File upload: {file.filename} ({file.content_type})")
    file_bytes = await file.read()
    result = _pipeline.ingest_file(file_bytes, file.filename or "upload")
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return IngestResponse(
        chunks_created=result["chunks_created"],
        source=result["source"],
        message=f"Ingested '{file.filename}' -> {result['chunks_created']} chunks stored."
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Basic metrics endpoint."""
    summary = query_logger.get_summary_metrics()
    return MetricsResponse(
        total_queries=summary["total_queries"],
        avg_latency_ms=summary["avg_latency_ms"],
        avg_faithfulness=0.0,
        avg_relevancy=0.0,
        top_sources=[]
    )


@router.get("/metrics/detail")
async def get_metrics_detail():
    """Full metrics including latency history and recent queries."""
    return query_logger.get_summary_metrics()


@router.get("/metrics/latency")
async def get_latency_history():
    """Latency over time for charting."""
    return query_logger.get_latency_history()


@router.delete("/collection")
async def reset_collection():
    """Wipe the vector store and start fresh."""
    logger.warning("Collection reset requested")
    _pipeline.reset()
    return {"message": "Collection wiped. All documents removed."}
