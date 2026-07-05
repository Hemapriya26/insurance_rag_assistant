"""
utils/suggestions.py
Phase 4 — Smart Suggested Questions.
IMPORTANT: this is a keyword/intent-based heuristic, not an LLM-generated
suggestion (no extra API call, zero added latency/cost). Suggestions are
templated per detected intent and filtered to avoid repeating the question
just asked.
"""

from typing import List

INTENT_FOLLOWUPS = {
    "Eligibility": ["What documents are needed to prove eligibility?", "Is there an age limit?"],
    "Benefits": ["What is excluded from this benefit?", "Is there a maximum claim amount?"],
    "Premium": ["Can the premium change after renewal?", "Are there any tax benefits on the premium?"],
    "Waiting Period": ["Does the waiting period apply to all conditions?", "Can the waiting period be waived?"],
    "Claim Process": ["What documents are required to file a claim?", "How long does claim settlement take?"],
    "Hospital Coverage": ["Are there non-network hospital options?", "Is there a room rent limit?"],
    "Exclusions": ["Are there any exceptions to this exclusion?", "What happens if I need excluded treatment?"],
    "Riders": ["What is the additional cost for this rider?", "Can riders be added after policy start?"],
    "Tax Benefits": ["Does this apply under the new tax regime?", "Is there a maximum deduction limit?"],
    "Renewal": ["What happens if I miss the renewal date?", "Does renewal require fresh underwriting?"],
    "Cancellation": ["Is there a refund if I cancel early?", "What is the notice period for cancellation?"],
    "Comparison": ["Which plan offers better hospital coverage?", "How do premiums compare over 5 years?"],
    "Summary": ["What are the key exclusions in this policy?", "What is the claim process for this policy?"],
    "General": ["What is the waiting period for this policy?", "What documents are needed to file a claim?"],
}


def generate_suggestions(intent: str, question_asked: str, limit: int = 2) -> List[str]:
    """Return up to `limit` follow-up question suggestions for the given intent,
    excluding anything identical to the question just asked."""
    candidates = INTENT_FOLLOWUPS.get(intent, INTENT_FOLLOWUPS["General"])
    filtered = [q for q in candidates if q.strip().lower() != question_asked.strip().lower()]
    return filtered[:limit]
