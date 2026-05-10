"""
frontend/app.py — Streamlit chat UI for the RAG system.
"""

import streamlit as st
import requests

st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000/api/v1"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Knowledge Base")
    st.divider()

    st.subheader("Add a URL or YouTube video")
    url_input = st.text_input("URL", placeholder="https://... or YouTube link")
    if st.button("Ingest URL", use_container_width=True):
        if url_input:
            with st.spinner("Ingesting..."):
                try:
                    res = requests.post(f"{API_BASE}/ingest/url", json={"url": url_input})
                    data = res.json()
                    st.success(f"✅ {data['message']}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a URL")

    st.divider()

    st.subheader("Upload a PDF or CSV")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv"])
    if uploaded_file and st.button("Ingest File", use_container_width=True):
        with st.spinner("Processing file..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_BASE}/ingest/file", files=files)
                data = res.json()
                st.success(f"✅ {data['message']}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    st.subheader("Options")
    use_web = st.toggle("Allow web search", value=False,
                        help="Let the agent search the web when docs don't have the answer")

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        requests.delete(f"{API_BASE}/collection")
        st.session_state.messages = []
        st.success("Knowledge base cleared!")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 Advanced RAG System")
st.caption("Hybrid retrieval · Agentic routing · Observability")

chat_tab, metrics_tab = st.tabs(["💬 Chat", "📊 Metrics"])

# ── Chat Tab ──────────────────────────────────────────────────────────────────
with chat_tab:

    # Display chat history
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

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(f"{API_BASE}/query", json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "use_web_search": use_web
                    })
                    data = res.json()

                    st.session_state.session_id = data.get("session_id")
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    latency = data.get("latency_ms", 0)

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 {len(sources)} sources"):
                            for i, src in enumerate(sources):
                                st.markdown(f"**Source {i+1}:** `{src['source']}`")
                                st.markdown(f"> {src['content'][:300]}...")

                    st.caption(f"⚡ {latency:.0f}ms")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": latency
                    })

                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Can't reach backend. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Metrics Tab ───────────────────────────────────────────────────────────────
with metrics_tab:
    st.subheader("System Observability Dashboard")

    if st.button("🔄 Refresh metrics"):
        st.rerun()

    try:
        res = requests.get(f"{API_BASE}/metrics/detail")
        m = res.json()
        total = m.get("total_queries", 0)

        if total == 0:
            st.info("No queries logged yet. Ask some questions in the Chat tab first!")
        else:
            # KPI row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Queries", total)
            col2.metric("Avg Latency", f"{m['avg_latency_ms']:.0f}ms")
            col3.metric("P99 Latency", f"{m['p99_latency_ms']:.0f}ms")
            col4.metric("Rewrite Rate", f"{m['rewrite_rate']*100:.0f}%",
                        help="% of queries where self-RAG rewrote the query")

            st.divider()

            # Route distribution + latency breakdown
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

            # Latency chart
            st.subheader("Latency Over Time")
            lat_res = requests.get(f"{API_BASE}/metrics/latency")
            lat_data = lat_res.json()
            if lat_data:
                import pandas as pd
                df = pd.DataFrame(lat_data)
                st.line_chart(df.set_index("index")["latency_ms"])

            st.divider()

            # Recent queries table
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

    except Exception as e:
        st.error(f"Could not load metrics: {e}")
