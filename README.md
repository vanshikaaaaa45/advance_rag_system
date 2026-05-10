---
title: Advanced RAG System
emoji: 🔍
colorFrom: purple
colorTo: teal
sdk: docker
app_file: frontend/app_hf.py
pinned: false
license: mit
short_description: Production RAG — hybrid retrieval, agents, observability
---

A production-grade Retrieval-Augmented Generation pipeline with hybrid retrieval, agentic query routing, self-correcting loops, and a real-time observability dashboard. Built entirely with free, local tools — no paid APIs required beyond Groq's free tier.

**Live demo:** [your-hf-spaces-link-here]

---

## What makes this different from a basic RAG

Most RAG tutorials build: PDF → embed → ask question → get answer.

This system builds:

| Feature             | Basic RAG          | This project                                       |
| ------------------- | ------------------ | -------------------------------------------------- |
| Retrieval           | Vector search only | Hybrid: dense + BM25 + RRF fusion                  |
| Reranking           | None               | Cross-encoder (ms-marco-MiniLM)                    |
| Query handling      | Fixed pipeline     | LangGraph agent with routing                       |
| Query improvement   | None               | HyDE query rewriting                               |
| Self-correction     | None               | Self-RAG loop (re-retrieves if quality is low)     |
| Hallucination check | None               | LLM-based grounding verification                   |
| Observability       | None               | P50/P99 latency, route distribution, JSONL logging |
| Sources             | Multiple formats   | PDF, URL, YouTube, CSV                             |

---

## Architecture

```
User query
    │
    ▼
LangGraph Agent
    ├── Route: direct answer (math, greetings — skips retrieval)
    ├── Route: vectorstore ──→ HyDE rewriting
    │                              │
    │                    ┌─────────┴──────────┐
    │                    ▼                    ▼
    │             Dense retrieval      BM25 sparse
    │          (sentence-transformers)  (rank-bm25)
    │                    └─────────┬──────────┘
    │                              ▼
    │                        RRF fusion
    │                              │
    │                              ▼
    │                   Cross-encoder reranking
    │                              │
    │                   Document grading ──→ Self-RAG loop
    │                   (filter irrelevant)   (rewrite + retry)
    │                              │
    │                              ▼
    │                      LLM generation
    │                    (Groq LLaMA 3.3 70B)
    │                              │
    │                              ▼
    │                   Hallucination check
    │                              │
    └── Route: web search          ▼
                             Response + citations
                             + conversation memory
```

---

## Tech stack (100% free)

| Component     | Tool                                 | Why                                        |
| ------------- | ------------------------------------ | ------------------------------------------ |
| LLM inference | Groq free tier                       | LLaMA 3.3 70B at 500 tok/s, no credit card |
| Embeddings    | sentence-transformers                | Local, free, 384-dim vectors               |
| Vector store  | ChromaDB                             | Persistent local vector DB                 |
| Sparse search | rank-bm25                            | BM25Okapi keyword search                   |
| Reranker      | cross-encoder/ms-marco-MiniLM-L-6-v2 | HuggingFace, local, free                   |
| Orchestration | LangChain + LangGraph                | Chains + stateful agent graph              |
| Backend       | FastAPI + uvicorn                    | Async REST API                             |
| Frontend      | Streamlit                            | Python-native UI                           |
| Hosting       | HuggingFace Spaces                   | Free deployment                            |

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/yourusername/advanced-rag
cd advanced-rag
./setup.sh
```

### 2. Add your Groq key

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

```bash
# Edit .env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Run

```bash
./run.sh
```

- Chat UI: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Project structure

```
advanced-rag/
├── backend/
│   ├── api/
│   │   ├── main.py          # FastAPI app, CORS, startup
│   │   └── routes.py        # /query /ingest /metrics endpoints
│   ├── ingestion/
│   │   ├── loaders.py       # PDF, URL, YouTube, CSV loaders
│   │   ├── chunker.py       # Recursive character text splitting
│   │   ├── embedder.py      # sentence-transformers + ChromaDB
│   │   └── pipeline.py      # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── dense.py         # Vector similarity search
│   │   ├── sparse.py        # BM25 keyword search
│   │   ├── fusion.py        # Reciprocal Rank Fusion
│   │   └── reranker.py      # Cross-encoder reranking
│   ├── agent/
│   │   ├── graph.py         # LangGraph state graph
│   │   ├── nodes.py         # Router, rewriter, grader, generator
│   │   └── memory.py        # Conversation history
│   ├── evaluation/
│   │   └── logger.py        # Query logging, P50/P99 metrics
│   ├── llm.py               # Groq/Ollama connector + RAG prompt
│   └── config.py            # Central settings
├── frontend/
│   └── app.py               # Streamlit chat + metrics dashboard
├── data/                    # Drop source documents here
├── chroma_db/               # Persisted vector store
├── setup.sh                 # First-time setup
├── run.sh                   # Start both servers
└── stop.sh                  # Stop all servers
```

---

## How the hybrid retrieval works

A query goes through 4 stages before reaching the LLM:

**1. HyDE rewriting** — instead of searching with the raw question, the agent generates a hypothetical answer first, then searches with that. The fake answer shares vocabulary with real answer chunks → better vector similarity.

**2. Dense retrieval** — embeds the rewritten query and finds the top-10 most semantically similar chunks in ChromaDB.

**3. BM25 sparse retrieval** — runs traditional keyword search across all stored chunks. Catches exact terms, acronyms, and proper nouns that semantic search misses.

**4. RRF + reranking** — Reciprocal Rank Fusion merges both result lists by rank (scale-invariant). A cross-encoder then scores each (query, chunk) pair together for final relevance ordering.

---

## How the self-RAG loop works

After retrieving and reranking, the agent grades each chunk for relevance. If fewer than 2 chunks are relevant:

1. The agent rewrites the query with a different HyDE hypothesis
2. Re-runs the full retrieval pipeline
3. Grades again

This loops up to 2 times. If a good answer is generated, a separate hallucination check verifies all claims are grounded in the retrieved context before returning to the user.

---

## Observability dashboard

The Metrics tab shows:

- **Total queries** and **avg/P50/P99 latency**
- **Rewrite rate** — how often the self-RAG loop triggered
- **Route distribution** — vectorstore vs direct vs web search
- **Latency over time** chart
- **Recent queries** table with per-query timing

Metrics persist across restarts in `logs/query_log.jsonl`.

---

## Configuration

All settings in `backend/config.py`:

```python
GROQ_MODEL = "llama-3.3-70b-versatile"   # LLM for generation
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512        # Characters per chunk
CHUNK_OVERLAP = 64      # Overlap between chunks
TOP_K_DENSE = 10        # Dense retrieval results
TOP_K_SPARSE = 10       # BM25 results
TOP_K_RERANK = 5        # Final chunks after reranking
RRF_K = 60              # RRF constant
```

---

## Running fully offline (no Groq)

Install Ollama and pull a model:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1
```

Then in `backend/config.py`:

```python
LLM_PROVIDER = "ollama"
```

No internet connection required for any component.

---

## Requirements

- Python 3.11
- ~2GB disk (models cached in `~/.cache/huggingface/`)
- 8GB RAM recommended
- Groq API key (free) or Ollama for fully local inference
