"""
agent/nodes.py — Individual node functions for the LangGraph agent.

WHAT IS A LANGGRAPH NODE?
In LangGraph, your pipeline is a directed graph where:
- NODES = functions that do work (retrieve, rewrite, grade, generate)
- EDGES = connections between nodes (what runs next)
- STATE = a shared dict that all nodes can read from and write to

Think of it like an assembly line:
  query → [router] → [retriever] → [grader] → [generator] → answer
                          ↑                          |
                          └──── [rewriter] ←─────────┘ (if answer is bad)

WHY AGENTS INSTEAD OF A FIXED PIPELINE?
Fixed pipeline (Phase 3): always does dense→sparse→fuse→rerank→generate.
Agent: DECIDES what to do based on the query and intermediate results.

Examples of agent decisions:
- "This question is about current events" → use web search, not vector DB
- "Retrieved chunks don't seem relevant" → rewrite the query and try again
- "Answer is well-supported by context" → return it
- "Answer seems hallucinated" → flag it and re-retrieve

This is called "Self-RAG" — the system evaluates its own outputs.

THE NODES WE BUILD:
1. route_query     → decides: retrieve from DB, web search, or answer from memory
2. rewrite_query   → HyDE: generates a better query for retrieval
3. retrieve        → runs the hybrid retrieval pipeline
4. grade_documents → scores whether retrieved docs are actually relevant
5. generate        → produces the final answer
6. grade_answer    → checks if answer is grounded (not hallucinated)
"""

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
import json

from backend.llm import get_llm, generate_answer
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.sparse import SparseRetriever
from backend.retrieval.fusion import RRFFusion
from backend.retrieval.reranker import CrossEncoderReranker

# Singletons — reused across node calls
_dense = DenseRetriever()
_sparse = SparseRetriever()
_fusion = RRFFusion()
_reranker = CrossEncoderReranker()


# ── Node 1: Query Router ──────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_template("""You are a query router for a RAG system.

Given a user question, decide which retrieval strategy to use:

- "vectorstore": Question can be answered from ingested documents (factual, specific, about the knowledge base content)
- "websearch": Question requires current/real-time information (news, today's events, live data)  
- "direct": Simple conversational question that doesn't need retrieval (greetings, math, general knowledge)

Respond with ONLY one word: vectorstore, websearch, or direct

Question: {question}
Decision:""")


def route_query(state: dict) -> dict:
    """
    Node 1: Decide where to route the query.

    WHAT IS ROUTING?
    Not every question needs vector DB retrieval.
    "What's 2+2?" → answer directly (no retrieval needed)
    "What happened in the news today?" → web search (vector DB has old data)
    "What does the document say about X?" → vector store retrieval

    The router saves tokens and latency by skipping unnecessary steps.

    STATE IN:  question, use_web_search
    STATE OUT: route ("vectorstore" | "websearch" | "direct")
    """
    question = state["question"]
    use_web_search = state.get("use_web_search", False)

    logger.info(f"Routing query: '{question[:60]}'")

    try:
        llm = get_llm(fast=True)  # Use fast model for routing — saves tokens
        chain = ROUTER_PROMPT | llm | StrOutputParser()
        route = chain.invoke({"question": question}).strip().lower()

        # Validate the route
        valid_routes = {"vectorstore", "websearch", "direct"}
        if route not in valid_routes:
            route = "vectorstore"  # Default fallback

        # Override: if user didn't enable web search, force vectorstore
        if route == "websearch" and not use_web_search:
            route = "vectorstore"
            logger.info("Web search disabled by user — routing to vectorstore")

        logger.info(f"Route decision: {route}")
        return {**state, "route": route}

    except Exception as e:
        logger.error(f"Router failed: {e} — defaulting to vectorstore")
        return {**state, "route": "vectorstore"}


# ── Node 2: Query Rewriter (HyDE) ─────────────────────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_template("""You are an expert at improving search queries for a RAG system.

TECHNIQUE: HyDE (Hypothetical Document Embeddings)
Instead of searching with the raw question, generate a SHORT hypothetical answer
that WOULD appear in a relevant document. This hypothetical answer, when embedded,
will be closer in vector space to the actual answer chunks.

Original question: {question}

Write a 2-3 sentence hypothetical document excerpt that would perfectly answer this question.
Be specific and use technical language appropriate for the topic.
Do NOT say "I" or "According to" — write it as if it's text FROM a document.

Hypothetical excerpt:""")


def rewrite_query(state: dict) -> dict:
    """
    Node 2: HyDE query rewriting.

    WHAT IS HyDE?
    Normal RAG: embed(question) → find similar chunks
    Problem: questions and answers have different vocabulary

    Question: "How does attention work?"
    Answer chunk: "The attention mechanism computes weighted sums of values..."

    The question vector ≠ the answer vector even though they're related.

    HyDE trick:
    1. Ask LLM: "Write a fake answer to this question"
    2. Embed the FAKE answer instead of the question
    3. Fake answer vector ≈ real answer vector (same vocabulary!)
    4. Find chunks similar to the fake answer → much better retrieval

    Research shows HyDE improves retrieval quality by 10-30%.

    STATE IN:  question, rewrite_count
    STATE OUT: rewritten_query (the HyDE-enhanced query)
    """
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    logger.info(f"HyDE rewriting query (attempt {rewrite_count + 1})")

    try:
        llm = get_llm(fast=True)
        chain = HYDE_PROMPT | llm | StrOutputParser()
        hypothetical_doc = chain.invoke({"question": question}).strip()

        logger.info(f"HyDE hypothesis: '{hypothetical_doc[:100]}...'")
        return {
            **state,
            "rewritten_query": hypothetical_doc,
            "rewrite_count": rewrite_count + 1
        }
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return {**state, "rewritten_query": question, "rewrite_count": rewrite_count + 1}


# ── Node 3: Retriever ──────────────────────────────────────────────────────────

def retrieve(state: dict) -> dict:
    """
    Node 3: Run hybrid retrieval pipeline.

    Uses the rewritten query if available, otherwise the original question.
    Runs: dense → sparse → RRF fusion → cross-encoder reranking

    STATE IN:  question, rewritten_query (optional)
    STATE OUT: documents (list of reranked Document objects)
    """
    # Use HyDE-rewritten query if available, else original
    query = state.get("rewritten_query") or state["question"]
    logger.info(f"Retrieving with query: '{query[:60]}'")

    dense_docs = _dense.retrieve(query)
    sparse_docs = _sparse.retrieve(query)
    fused_docs = _fusion.fuse(dense_docs, sparse_docs)
    reranked_docs = _reranker.rerank(state["question"], fused_docs)

    logger.info(f"Retrieved {len(reranked_docs)} documents")
    return {**state, "documents": reranked_docs}


# ── Node 4: Document Grader ────────────────────────────────────────────────────

GRADER_PROMPT = ChatPromptTemplate.from_template("""You are grading whether a document is relevant to a question.

Question: {question}
Document: {document}

Is this document relevant to answering the question?
Respond with ONLY: yes or no

Answer:""")


def grade_documents(state: dict) -> dict:
    """
    Node 4: Filter retrieved documents for relevance.

    WHY GRADE DOCUMENTS?
    The retriever returns the TOP-K most similar chunks, but similarity
    doesn't guarantee relevance. A chunk might be topically related but
    not actually helpful for answering the specific question.

    The grader uses the LLM to judge each chunk:
    - "yes" → keep it
    - "no" → discard it

    If too many chunks are discarded → trigger query rewrite + re-retrieval.

    STATE IN:  question, documents
    STATE OUT: documents (filtered), needs_rewrite (bool)
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {**state, "documents": [], "needs_rewrite": True}

    logger.info(f"Grading {len(documents)} documents for relevance")

    llm = get_llm(fast=True)
    chain = GRADER_PROMPT | llm | StrOutputParser()

    relevant_docs = []
    for doc in documents:
        try:
            verdict = chain.invoke({
                "question": question,
                "document": doc.page_content[:500]  # First 500 chars enough to judge
            }).strip().lower()

            if "yes" in verdict:
                relevant_docs.append(doc)
                logger.info(f"  ✓ Relevant: '{doc.page_content[:60]}...'")
            else:
                logger.info(f"  ✗ Irrelevant: '{doc.page_content[:60]}...'")
        except Exception as e:
            logger.error(f"Grading failed for chunk: {e}")
            relevant_docs.append(doc)  # Keep on error (fail open)

    # If less than 2 relevant docs found → rewrite query and try again
    needs_rewrite = len(relevant_docs) < 2
    logger.info(
        f"Grading complete: {len(relevant_docs)}/{len(documents)} relevant. "
        f"Needs rewrite: {needs_rewrite}"
    )

    return {**state, "documents": relevant_docs, "needs_rewrite": needs_rewrite}


# ── Node 5: Generator ──────────────────────────────────────────────────────────

def generate(state: dict) -> dict:
    """
    Node 5: Generate the final answer.

    Uses the filtered, graded documents as context.
    Falls back gracefully if no relevant documents found.

    STATE IN:  question, documents
    STATE OUT: answer, generation_count
    """
    question = state["question"]
    documents = state.get("documents", [])
    generation_count = state.get("generation_count", 0)

    logger.info(f"Generating answer from {len(documents)} documents")

    if not documents:
        answer = (
            "I couldn't find relevant information in the knowledge base to answer "
            "your question. Try ingesting more documents or rephrasing your question."
        )
    else:
        answer = generate_answer(question, documents)

    return {**state, "answer": answer, "generation_count": generation_count + 1}


# ── Node 6: Answer Grader (hallucination check) ───────────────────────────────

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template("""You are checking if an answer is grounded in the provided documents.

Documents:
{documents}

Answer: {answer}

Is the answer supported by the documents? (Does it only use information from the documents?)
Respond with ONLY: yes or no

Answer:""")


def grade_answer(state: dict) -> dict:
    """
    Node 6: Check if the generated answer is grounded in the documents.

    WHY CHECK FOR HALLUCINATIONS?
    LLMs sometimes "fill in gaps" with plausible-sounding but made-up facts,
    even when instructed not to. This node catches those cases.

    If the answer contains claims not in the documents → regenerate.
    We limit regeneration attempts to avoid infinite loops.

    STATE IN:  question, documents, answer, generation_count
    STATE OUT: is_grounded (bool)
    """
    documents = state.get("documents", [])
    answer = state.get("answer", "")
    generation_count = state.get("generation_count", 0)

    # Don't check if we've already tried twice (avoid infinite loop)
    if generation_count >= 2 or not documents:
        return {**state, "is_grounded": True}

    logger.info("Checking answer for hallucinations...")

    try:
        llm = get_llm(fast=True)
        chain = HALLUCINATION_PROMPT | llm | StrOutputParser()

        doc_text = "\n\n".join([d.page_content[:300] for d in documents[:3]])
        verdict = chain.invoke({
            "documents": doc_text,
            "answer": answer
        }).strip().lower()

        is_grounded = "yes" in verdict
        logger.info(f"Hallucination check: {'grounded ✓' if is_grounded else 'hallucinated ✗'}")
        return {**state, "is_grounded": is_grounded}

    except Exception as e:
        logger.error(f"Answer grading failed: {e}")
        return {**state, "is_grounded": True}  # Fail open


# ── Node 7: Direct Answer (no retrieval needed) ───────────────────────────────

DIRECT_PROMPT = ChatPromptTemplate.from_template("""You are a helpful AI assistant. Answer this question directly and concisely.

Question: {question}

Answer:""")


def direct_answer(state: dict) -> dict:
    """
    Node 7: Answer directly without retrieval (for simple questions).
    Used when router decides the question doesn't need document retrieval.
    """
    question = state["question"]
    logger.info(f"Answering directly (no retrieval): '{question[:60]}'")

    try:
        llm = get_llm(fast=False)
        chain = DIRECT_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        return {**state, "answer": answer, "documents": []}
    except Exception as e:
        logger.error(f"Direct answer failed: {e}")
        return {**state, "answer": f"Error: {e}", "documents": []}
