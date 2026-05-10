"""
frontend/app_hf.py — HuggingFace Spaces version of the app.

KEY DIFFERENCE from app.py:
On HF Spaces we can't run FastAPI separately.
This version calls the RAG pipeline DIRECTLY from Streamlit,
cutting out the HTTP layer entirely.

GROQ API KEY on HF Spaces:
Go to your Space → Settings → Repository secrets → Add GROQ_API_KEY
HF Spaces injects it as an environment variable automatically.
"""

import streamlit as st
import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Lazy load pipeline (cached so it only loads once) ─────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline... (first load takes ~30s)")
def load_pipeline():
    """
    Load all pipeline components once and cache them.
    st.cache_resource persists across reruns for the lifetime of the app.
    This means models are loaded once, not on every user interaction.
    """
    from backend.ingestion.pipeline import IngestionPipeline
    from backend.agent.graph import get_graph
    from backend.agent.memory import memory
    from backend.evaluation.logger import query_logger

    pipeline = IngestionPipeline()
    graph = get_graph()

    return {
        "pipeline": pipeline,
        "graph": graph,
        "memory": memory,
        "logger": query_logger,
    }


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{int(time.time())}"

# ── Load pipeline ─────────────────────────────────────────────────────────────
try:
    components = load_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    st.error(f"Pipeline failed to load: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Knowledge Base")
    st.divider()

    st.subheader("Add a URL or YouTube video")
    url_input = st.text_input("URL", placeholder="https://... or YouTube link")
    if st.button("Ingest URL", use_container_width=True):
        if url_input and pipeline_ready:
            with st.spinner("Ingesting..."):
                try:
                    result = components["pipeline"].ingest_url(url_input)
                    if result["success"]:
                        st.success(f"✅ Ingested {result['docs_loaded']} pages → {result['chunks_created']} chunks stored.")
                    else:
                        st.error(f"Failed: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error: {e}")
        elif not url_input:
            st.warning("Please enter a URL")

    st.divider()

    st.subheader("Upload a PDF or CSV")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv"])
    if uploaded_file and st.button("Ingest File", use_container_width=True):
        with st.spinner("Processing file..."):
            try:
                result = components["pipeline"].ingest_file(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                if result["success"]:
                    st.success(f"✅ Ingested '{uploaded_file.name}' → {result['chunks_created']} chunks stored.")
                else:
                    st.error(f"Failed: {result.get('error')}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    st.subheader("Options")
    use_web = st.toggle("Allow web search", value=False)

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        if pipeline_ready:
            components["pipeline"].reset()
            st.session_state.messages = []
            st.success("Knowledge base cleared!")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 Advanced RAG System")
st.caption("Hybrid retrieval · Agentic routing · Observability")

chat_tab, metrics_tab = st.tabs(["💬 Chat", "📊 Metrics"])

# ── Chat Tab ──────────────────────────────────────────────────────────────────
with chat_tab:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} sources", expanded=False):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**Source {i+1}:** `{src['source']}`")
                        st.markdown(f"> {src['content'][:300]}...")
                        st.caption(f"Relevance score: {src['score']:.3f}")

            if msg.get("latency_ms"):
                st.caption(f"⚡ {msg['latency_ms']:.0f}ms")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if not pipeline_ready:
                    st.error("Pipeline not loaded. Check your GROQ_API_KEY secret.")
                else:
                    try:
                        start = time.time()
                        graph = components["graph"]
                        memory = components["memory"]
                        logger = components["logger"]

                        initial_state = {
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                            "use_web_search": use_web,
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

                        memory.add_turn(st.session_state.session_id, prompt, answer)

                        logger.log_query(
                            question=prompt,
                            answer=answer,
                            route=final_state.get("route", "unknown"),
                            latency_ms=latency_ms,
                            num_docs=len(docs),
                            rewrite_count=final_state.get("rewrite_count", 0),
                        )

                        sources = [
                            {
                                "content": doc.page_content[:300],
                                "source": doc.metadata.get("source", "Unknown"),
                                "score": float(doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0.0)))
                            }
                            for doc in docs
                        ]

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📎 {len(sources)} sources"):
                                for i, src in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:** `{src['source']}`")
                                    st.markdown(f"> {src['content'][:300]}...")

                        st.caption(f"⚡ {latency_ms:.0f}ms")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "latency_ms": latency_ms
                        })

                    except Exception as e:
                        st.error(f"Error: {e}")

# ── Metrics Tab ───────────────────────────────────────────────────────────────
with metrics_tab:
    st.subheader("System Observability Dashboard")

    if st.button("🔄 Refresh metrics"):
        st.rerun()

    if not pipeline_ready:
        st.warning("Pipeline not loaded.")
    else:
        m = components["logger"].get_summary_metrics()
        total = m.get("total_queries", 0)

        if total == 0:
            st.info("No queries logged yet. Ask some questions in the Chat tab first!")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Queries", total)
            col2.metric("Avg Latency", f"{m['avg_latency_ms']:.0f}ms")
            col3.metric("P99 Latency", f"{m['p99_latency_ms']:.0f}ms")
            col4.metric("Rewrite Rate", f"{m['rewrite_rate']*100:.0f}%")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Route Distribution")
                routes = m.get("route_distribution", {})
                for route, count in routes.items():
                    pct = count / total * 100
                    st.markdown(f"**{route}** — {count} queries ({pct:.0f}%)")
                    st.progress(pct / 100)

            with col2:
                st.subheader("Latency Breakdown")
                st.markdown(f"**Min:** {m['min_latency_ms']:.0f}ms")
                st.markdown(f"**P50:** {m['p50_latency_ms']:.0f}ms")
                st.markdown(f"**P99:** {m['p99_latency_ms']:.0f}ms")
                st.markdown(f"**Max:** {m['max_latency_ms']:.0f}ms")

            st.divider()

            st.subheader("Latency Over Time")
            lat_data = components["logger"].get_latency_history()
            if lat_data:
                import pandas as pd
                df = pd.DataFrame(lat_data)
                st.line_chart(df.set_index("index")["latency_ms"])

            st.divider()

            st.subheader("Recent Queries")
            recent = m.get("recent_queries", [])
            if recent:
                import pandas as pd
                df = pd.DataFrame([{
                    "Question": q["question"][:60] + "...",
                    "Route": q["route"],
                    "Latency (ms)": q["latency_ms"],
                } for q in reversed(recent)])
                st.dataframe(df, use_container_width=True)