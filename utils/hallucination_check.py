"""
utils/hallucination_check.py
Phase 3, Module 6 — Hallucination Detection.
Lightweight groundedness heuristic (word-overlap between answer sentences
and retrieved context), NOT a trained NLI/entailment model. It catches the
clearest cases (an answer introducing content absent from any retrieved
chunk) without adding another LLM call. Documented as a heuristic, not a
guarantee.
"""

import re
from typing import List, Dict, Any


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check_groundedness(answer: str, context: str, min_overlap: float = 0.25) -> Dict[str, Any]:
    """Return groundedness verdict + per-sentence overlap ratios."""
    if answer.strip() == "Information not found in documents.":
        return {"grounded": True, "overlap_ratio": 1.0, "flagged_sentences": []}

    context_tokens = set(context.lower().split())
    flagged = []
    ratios = []

    for sentence in _sentences(answer):
        sent_tokens = set(re.findall(r"[a-zA-Z0-9%]+", sentence.lower()))
        sent_tokens = {t for t in sent_tokens if len(t) > 3}  # ignore stopword-ish noise
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        ratios.append(overlap)
        if overlap < min_overlap:
            flagged.append(sentence)

    avg_overlap = round(sum(ratios) / len(ratios), 3) if ratios else 1.0
    return {
        "grounded": len(flagged) == 0,
        "overlap_ratio": avg_overlap,
        "flagged_sentences": flagged,
    }
