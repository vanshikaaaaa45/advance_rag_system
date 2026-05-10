"""
agent/graph.py — LangGraph state graph definition.

WHAT IS A STATE GRAPH?
A StateGraph is a directed graph where:
- Each NODE is a Python function that transforms the state
- Each EDGE is a connection saying "after node A, go to node B"
- CONDITIONAL EDGES say "after node A, go to B or C depending on state"

THE GRAPH WE BUILD:

                    ┌─────────────┐
                    │ route_query │
                    └──────┬──────┘
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     [vectorstore]     [direct]      [websearch]
            │              │              │
     rewrite_query    direct_answer   direct_answer
            │
         retrieve
            │
      grade_documents
            │
    ┌───────┴───────┐
    │ needs_rewrite?│
    └───────┬───────┘
    NO ─────┤───── YES (max 2 times)
            │              │
         generate      rewrite_query → retrieve (loop back)
            │
      grade_answer
            │
    ┌───────┴───────┐
    │  is_grounded? │
    └───────┬───────┘
    YES ────┤───── NO (max 2 generations)
            │              │
           END          generate (retry)

KEY CONCEPTS:

1. CONDITIONAL EDGES:
   After grade_documents, we check state["needs_rewrite"].
   If True → go back to rewrite_query (self-correcting loop)
   If False → go to generate

2. LOOP PREVENTION:
   rewrite_count and generation_count prevent infinite loops.
   Max 2 rewrites, max 2 generations.

3. STATE TYPING:
   We use TypedDict to define exactly what keys the state has.
   This catches bugs early and makes the code self-documenting.
"""

from typing import TypedDict, Annotated
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from loguru import logger

from backend.agent.nodes import (
    route_query,
    rewrite_query,
    retrieve,
    grade_documents,
    generate,
    grade_answer,
    direct_answer,
)


# ── State Schema ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """
    The shared state passed between all nodes.

    Every node receives this dict and returns an updated version.
    LangGraph merges the updates automatically.
    """
    question: str                    # Original user question
    session_id: str                  # For multi-turn conversations
    use_web_search: bool             # User preference
    route: str                       # Router decision
    rewritten_query: str             # HyDE-enhanced query
    documents: list[Document]        # Retrieved + filtered documents
    answer: str                      # Generated answer
    rewrite_count: int               # How many times we've rewritten
    generation_count: int            # How many times we've generated
    needs_rewrite: bool              # Grader's verdict
    is_grounded: bool                # Hallucination check result


# ── Conditional Edge Functions ────────────────────────────────────────────────

def decide_after_routing(state: RAGState) -> str:
    """
    After routing, decide which path to take.
    Returns the name of the next node to visit.
    """
    route = state.get("route", "vectorstore")
    logger.info(f"Edge: route={route}")

    if route == "direct":
        return "direct_answer"
    elif route == "websearch":
        return "direct_answer"  # Fallback to direct for now (web search in Phase 4+)
    else:
        return "rewrite_query"  # vectorstore path → rewrite first


def decide_after_grading(state: RAGState) -> str:
    """
    After grading documents, decide whether to rewrite or generate.
    """
    needs_rewrite = state.get("needs_rewrite", False)
    rewrite_count = state.get("rewrite_count", 0)

    # Cap rewrites at 2 to prevent infinite loops
    if needs_rewrite and rewrite_count < 2:
        logger.info(f"Edge: needs_rewrite=True, attempt {rewrite_count}/2 → rewrite_query")
        return "rewrite_query"
    else:
        logger.info("Edge: proceeding to generate")
        return "generate"


def decide_after_answer_grading(state: RAGState) -> str:
    """
    After hallucination check, decide whether to return or regenerate.
    """
    is_grounded = state.get("is_grounded", True)
    generation_count = state.get("generation_count", 0)

    if not is_grounded and generation_count < 2:
        logger.info(f"Edge: hallucination detected, regenerating (attempt {generation_count}/2)")
        return "generate"
    else:
        logger.info("Edge: answer is grounded → END")
        return END


# ── Build the Graph ────────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Construct and compile the full RAG agent graph.

    Returns a compiled graph ready to invoke with:
        graph.invoke({"question": "...", ...})
    """
    # Create the graph with our state schema
    graph = StateGraph(RAGState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("route_query", route_query)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("direct_answer", direct_answer)

    # ── Set entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("route_query")

    # ── Add edges ─────────────────────────────────────────────────────────────

    # After routing → branch based on route decision
    graph.add_conditional_edges(
        "route_query",
        decide_after_routing,
        {
            "rewrite_query": "rewrite_query",
            "direct_answer": "direct_answer",
        }
    )

    # Direct answer path → END
    graph.add_edge("direct_answer", END)

    # Vectorstore path: rewrite → retrieve → grade
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # After grading → rewrite again OR generate
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        }
    )

    # After generating → check for hallucinations
    graph.add_edge("generate", "grade_answer")

    # After hallucination check → regenerate OR end
    graph.add_conditional_edges(
        "grade_answer",
        decide_after_answer_grading,
        {
            "generate": "generate",
            END: END,
        }
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("RAG agent graph compiled successfully")
    return compiled


# ── Singleton graph instance ──────────────────────────────────────────────────
# Built once at import time, reused for all requests
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph
