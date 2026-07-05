"""
utils/evaluation.py
Phase 3, Module 9 — Evaluation Framework.
IMPORTANT: these are lightweight, dependency-free HEURISTIC approximations
of RAG evaluation metrics (word-overlap based), not the trained/LLM-judge
implementations used by frameworks like RAGAS. They are useful as a quick
directional signal, not a certified accuracy benchmark. Swap in RAGAS if a
real evaluation dependency budget is available.
"""

import re
from typing import Dict, Any, List


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-zA-Z0-9%]+", text.lower()) if len(t) > 3}


def evaluate_response(question: str, answer: str, context: str) -> Dict[str, Any]:
    q_tokens, a_tokens, c_tokens = _tokens(question), _tokens(answer), _tokens(context)

    faithfulness = len(a_tokens & c_tokens) / len(a_tokens) if a_tokens else 1.0
    answer_relevance = len(a_tokens & q_tokens) / len(q_tokens) if q_tokens else 0.0
    context_precision = len(c_tokens & q_tokens) / len(c_tokens) if c_tokens else 0.0
    context_recall = len(q_tokens & c_tokens) / len(q_tokens) if q_tokens else 0.0
    groundedness = faithfulness
    correctness_proxy = (faithfulness + answer_relevance) / 2

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_precision": round(context_precision, 3),
        "context_recall": round(context_recall, 3),
        "groundedness": round(groundedness, 3),
        "correctness_proxy": round(correctness_proxy, 3),
        "method": "heuristic-word-overlap (not RAGAS)",
    }


def evaluation_report(evaluations: List[Dict[str, Any]]) -> str:
    if not evaluations:
        return "No evaluated responses yet."
    keys = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "groundedness"]
    lines = ["# Evaluation Report (heuristic metrics)\n"]
    for k in keys:
        avg = round(sum(e[k] for e in evaluations) / len(evaluations), 3)
        lines.append(f"- **{k.replace('_', ' ').title()}**: {avg}")
    lines.append(f"\n_Based on {len(evaluations)} responses. Method: word-overlap heuristic, not a trained judge model._")
    return "\n".join(lines)
