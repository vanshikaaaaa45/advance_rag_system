"""
config.py — Central configuration for the entire RAG project.

WHY: Instead of hardcoding model names, paths, and settings scattered
across every file, we keep them all here. Change once → updates everywhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Reads your .env file automatically

# ── Project Paths ────────────────────────────────────────────────────────────
# Path(__file__) = this file's location
# .parent = the backend/ folder
# .parent.parent = the project root (advanced-rag/)
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"          # Where you drop your source documents
CHROMA_DIR = ROOT_DIR / "chroma_db"  # Where ChromaDB persists the vectors

# ── LLM Settings ─────────────────────────────────────────────────────────────
# We use Groq for generation — it's free and blazing fast (LLaMA 3 70B)
# Switch to "ollama" if you want to run 100% offline
LLM_PROVIDER = "groq"  # "groq" or "ollama"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile" # Best free model on Groq
GROQ_FAST_MODEL = "llama-3.1-8b-instant" # Faster, used for routing decisions

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1"

# ── Embedding Settings ────────────────────────────────────────────────────────
# sentence-transformers runs locally — completely free, no API needed.
# all-MiniLM-L6-v2: fast, small (80MB), 384 dimensions — great for dev.
# all-mpnet-base-v2: slower, larger (420MB), 768 dims — better quality.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Chunking Settings ─────────────────────────────────────────────────────────
# CHUNK_SIZE: how many characters per chunk
# Why not too large? LLM context is limited. Why not too small? Loses context.
# 512 chars ≈ ~128 tokens — a good middle ground.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64   # Overlap between chunks so we don't cut sentences mid-thought

# ── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K_DENSE = 10    # How many results to fetch from vector DB
TOP_K_SPARSE = 10   # How many results BM25 returns
TOP_K_RERANK = 5    # After fusion + reranking, keep top 5 for generation
RRF_K = 60          # RRF constant — 60 is the standard, rarely needs changing

# ── Reranker ──────────────────────────────────────────────────────────────────
# Cross-encoder scores PAIRS of (query, chunk) — much more accurate than
# embedding similarity alone, but slower. We only run it on the fused top-K.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_documents"

# ── LangSmith Observability ───────────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "advanced-rag")

# ── Evaluation ────────────────────────────────────────────────────────────────
# RAGAs needs an LLM to judge quality. We use Groq so it's free.
EVAL_SAMPLE_SIZE = 5  # How many Q&A pairs to evaluate at once (keep low for speed)
