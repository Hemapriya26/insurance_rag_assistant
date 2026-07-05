"""
utils/reranker.py
Phase 3, Module 3 — Re-ranking.
Uses a real cross-encoder (sentence-transformers) when available. Because
that dependency pulls in torch and can be multi-GB, it is OPTIONAL: if it
isn't installed, this module falls back to a transparent lexical-overlap
heuristic so the app still runs and still improves ordering over raw RRF.
The active mode is surfaced in the Admin Dashboard so it's never silent.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

_cross_encoder = None
_CROSS_ENCODER_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _CROSS_ENCODER_AVAILABLE = True
    logger.info("Cross-encoder reranker loaded (ms-marco-MiniLM-L-6-v2)")
except Exception as exc:  # ImportError or model download failure
    logger.warning("Cross-encoder unavailable (%s) — using lexical-overlap fallback", exc)


def reranker_mode() -> str:
    return "cross-encoder" if _CROSS_ENCODER_AVAILABLE else "lexical-overlap (fallback)"


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def rerank(query: str, candidates: List[Tuple[Document, float]], top_n: int = None) -> List[dict]:
    """
    Re-rank fused (doc, rrf_score) candidates and return the top_n with full
    scoring transparency: original RRF score, rerank score, and final rank.
    """
    top_n = top_n or CONFIG.retrieval.top_k_final

    if _CROSS_ENCODER_AVAILABLE:
        pairs = [[query, doc.page_content] for doc, _ in candidates]
        rerank_scores = _cross_encoder.predict(pairs).tolist()
    else:
        rerank_scores = [_lexical_overlap_score(query, doc.page_content) for doc, _ in candidates]

    scored = [
        {"document": doc, "rrf_score": rrf_score, "rerank_score": round(float(rs), 4)}
        for (doc, rrf_score), rs in zip(candidates, rerank_scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    for i, item in enumerate(scored[:top_n], 1):
        item["final_rank"] = i

    return scored[:top_n]
