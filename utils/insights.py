"""
utils/insights.py
Phase 5 — Insights Dashboard rendering. All four sections reuse data that
Phase 3/4 already silently collects in st.session_state (query_log,
evaluations) — this page is purely a new display surface, no new data
collection logic.
"""

import os
from typing import List, Dict, Any, Optional
import streamlit as st
from utils.analytics import compute_kpis, to_dataframe, daily_query_counts
from utils.evaluation import evaluation_report
from utils.stats import session_summary


def render_analytics_section(query_log: List[Dict[str, Any]]) -> None:
    if not query_log:
        st.info("Ask a few questions in Chat to populate analytics.")
        return

    kpis = compute_kpis(query_log)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Questions", kpis["total_questions"])
    c2.metric("Avg Confidence", kpis["avg_confidence_score"])
    c3.metric("Avg Retrieval Time", f"{kpis['avg_retrieval_time']}s")
    c4.metric("Avg Generation Time", f"{kpis['avg_generation_time']}s")

    df = to_dataframe(query_log)
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Provider Usage")
        st.bar_chart(df["provider"].value_counts())
    with col_b:
        if "intent" in df:
            st.caption("Intent Distribution")
            st.bar_chart(df["intent"].value_counts())

    daily = daily_query_counts(query_log)
    if daily:
        st.caption("Daily Usage")
        st.bar_chart(daily)


def render_evaluation_section(evaluations: List[Dict[str, Any]]) -> None:
    if not evaluations:
        st.info("Ask a few questions in Chat to populate evaluation data.")
        return

    keys = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "groundedness"]
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        avg = round(sum(e[key] for e in evaluations) / len(evaluations), 3)
        col.metric(key.replace("_", " ").title(), avg)

    st.caption("Method: word-overlap heuristic (not a trained judge model or RAGAS) — a directional signal, not a certified score.")

    with st.expander("Evaluation History"):
        for i, ev in enumerate(evaluations[-20:], 1):
            st.caption(f"#{i}: faithfulness={ev['faithfulness']}, relevance={ev['answer_relevance']}, groundedness={ev['groundedness']}")

    report = evaluation_report(evaluations)
    st.download_button("⬇️ Download Evaluation Report", report, "evaluation_report.md", "text/markdown")


def render_performance_section(query_log: List[Dict[str, Any]], memory) -> None:
    summary = session_summary(query_log)
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Retrieval Latency", f"{summary['avg_retrieval']}s")
    c2.metric("Avg Generation Latency", f"{summary['avg_generation']}s")
    c3.metric("Total Queries This Session", summary["total_queries"])

    c4, c5 = st.columns(2)
    c4.metric("Conversation Turns in Memory", len(memory.turns))
    c5.metric("Memory Summarized", "Yes" if memory.summary else "No")

    if query_log:
        avg_context_chunks = "n/a"
        st.caption(f"Session covers {len(query_log)} exchanges.")


def render_knowledge_base_section(
    vectorstore, bm25_index, chunks: list, index_dir: str,
) -> None:
    from utils.document_intelligence import document_statistics

    if not chunks or vectorstore is None:
        st.info("Build the knowledge base first to see these metrics.")
        return

    stats = document_statistics(chunks)
    total_pages = sum(s["pages"] for s in stats.values() if isinstance(s["pages"], int))
    index_size_bytes = 0
    if os.path.isdir(index_dir):
        index_size_bytes = sum(
            os.path.getsize(os.path.join(index_dir, f)) for f in os.listdir(index_dir)
            if os.path.isfile(os.path.join(index_dir, f))
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDFs", len(stats))
    c2.metric("Pages", total_pages or "n/a")
    c3.metric("Chunks", len(chunks))
    c4.metric("Embeddings", len(chunks))

    c5, c6, c7 = st.columns(3)
    c5.metric("Index Size", f"{index_size_bytes / 1024:.1f} KB")
    c6.metric("FAISS Status", "🟢 Active")
    c7.metric("Hybrid Search", "🟢 Active" if bm25_index else "⚪ FAISS-only")
