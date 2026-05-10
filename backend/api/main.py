"""
main.py — The entry point for the entire backend.

FastAPI is a modern Python web framework. When you run this file,
it starts an HTTP server. Your Streamlit frontend talks to this server.

WHY FastAPI over Flask?
- Automatic API docs at /docs (try it in browser after running)
- Built-in request validation via Pydantic
- Async support — handles multiple requests without blocking
- Faster and more modern

To run: uvicorn backend.api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from backend.api.routes import router
from backend.config import LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT

# ── Configure logging ────────────────────────────────────────────────────────
# loguru is a drop-in replacement for Python's built-in logging.
# It's cleaner and colorized in terminal — great for debugging.
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level="INFO"
)
logger.add(
    "logs/app.log",      # Also write to a file for history
    rotation="10 MB",    # Start a new log file when it hits 10MB
    retention="7 days",  # Delete logs older than 7 days
    level="DEBUG"
)

# ── Create FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="Advanced RAG System",
    description="Production-grade RAG with hybrid retrieval, agentic routing, and evaluation",
    version="1.0.0",
    docs_url="/docs",   # Visit http://localhost:8000/docs to see all endpoints
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
# Without this, your browser blocks requests from Streamlit (port 8501)
# to FastAPI (port 8000) because they're on different ports = "different origins".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, lock this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
# All the actual endpoints (/query, /ingest, /metrics) live in routes.py
# We use a "router" pattern to keep main.py clean.
app.include_router(router, prefix="/api/v1")


# ── Startup event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Runs once when the server starts."""
    logger.info("Advanced RAG System starting up...")
    logger.info(f"LangSmith tracing: {LANGCHAIN_TRACING_V2}")
    logger.info(f"LangSmith project: {LANGCHAIN_PROJECT}")
    logger.info("Server ready. Visit http://localhost:8000/docs")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple endpoint to verify the server is running."""
    return {"status": "ok", "message": "Advanced RAG System is running"}


# ── Run directly ──────────────────────────────────────────────────────────────
# This block runs only when you do: python -m backend.api.main
# (Not when uvicorn imports it — uvicorn handles its own startup)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
