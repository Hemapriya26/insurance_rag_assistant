"""
stats.py
Phase 2.5 — lightweight, non-invasive statistics helpers (word count, rough
token estimate, session aggregates). No backend logic touched.
"""

from typing import List, Dict, Any


def word_count(text: str) -> int:
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Rough heuristic: ~0.75 words per token on average for English text."""
    return max(1, round(word_count(text) / 0.75))


def session_summary(query_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not query_log:
        return {"total_queries": 0, "avg_retrieval": 0.0, "avg_generation": 0.0}

    total = len(query_log)
    avg_retrieval = round(sum(q["retrieval_time"] for q in query_log) / total, 2)
    avg_generation = round(sum(q["generation_time"] for q in query_log) / total, 2)

    return {
        "total_queries": total,
        "avg_retrieval": avg_retrieval,
        "avg_generation": avg_generation,
    }
