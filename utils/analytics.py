"""
utils/analytics.py
Phase 3, Module 7 — Analytics Dashboard aggregation logic.
Operates purely on st.session_state.query_log (in-memory, per-session).
"""

from collections import Counter
from typing import List, Dict, Any
import pandas as pd


def compute_kpis(query_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not query_log:
        return {
            "total_questions": 0, "avg_retrieval_time": 0.0, "avg_generation_time": 0.0,
            "avg_confidence_score": 0.0, "provider_usage": {}, "model_usage": {},
            "most_asked_intents": {},
        }

    df = pd.DataFrame(query_log)
    confidence_map = {"High": 1.0, "Medium": 0.6, "Low": 0.2}
    df["confidence_score"] = df["confidence"].map(confidence_map).fillna(0.2)

    return {
        "total_questions": len(df),
        "avg_retrieval_time": round(df["retrieval_time"].mean(), 2),
        "avg_generation_time": round(df["generation_time"].mean(), 2),
        "avg_confidence_score": round(df["confidence_score"].mean(), 2),
        "provider_usage": df["provider"].value_counts().to_dict(),
        "model_usage": df["model"].value_counts().to_dict(),
        "most_asked_intents": dict(Counter(df.get("intent", pd.Series(dtype=str))).most_common(5)),
    }


def to_dataframe(query_log: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(query_log)


def daily_query_counts(query_log: List[Dict[str, Any]]) -> Dict[str, int]:
    """Phase 4 addition — query count per calendar day, for a trend chart.
    Requires each query_log entry to have a 'timestamp' field (ISO date/time
    string); entries without one are skipped rather than raising."""
    counts: Dict[str, int] = {}
    for entry in query_log:
        ts = entry.get("timestamp")
        if not ts:
            continue
        day = ts.split("T")[0] if "T" in ts else ts.split(" ")[0]
        counts[day] = counts.get(day, 0) + 1
    return counts
