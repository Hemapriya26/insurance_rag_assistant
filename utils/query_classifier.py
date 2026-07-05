"""
utils/query_classifier.py
Phase 3, Module 4 — Intelligent Query Understanding.
Keyword-based intent classifier (no extra LLM call, zero added latency/cost).
This is a heuristic, not a trained classifier — documented as such.
"""

from typing import Dict, List

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "Eligibility": ["eligible", "eligibility", "who can apply", "qualify"],
    "Benefits": ["benefit", "coverage", "covered", "cover "],
    "Premium": ["premium", "cost", "price", "fee"],
    "Waiting Period": ["waiting period", "wait time", "cooling"],
    "Claim Process": ["claim", "reimbursement", "how to file"],
    "Hospital Coverage": ["hospital", "network hospital", "cashless"],
    "Exclusions": ["exclusion", "not covered", "excluded"],
    "Riders": ["rider", "add-on", "add on"],
    "Tax Benefits": ["tax", "80d", "deduction"],
    "Renewal": ["renew", "renewal"],
    "Cancellation": ["cancel", "cancellation", "terminate"],
    "Comparison": ["compare", "versus", " vs ", "difference between"],
    "Summary": ["summarize", "summary", "overview"],
}


def classify_intent(question: str) -> str:
    q = f" {question.lower()} "
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "General"
