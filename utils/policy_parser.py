"""
utils/policy_parser.py
Phase 5 — Policy Summary Generator + Clause Extraction (Document Intelligence).
These are user-triggered (button click on the Document Intelligence page),
not run automatically per chat turn — each call is one LLM invocation per
document, so cost/latency is bounded and predictable.
"""

from typing import Dict, Any, List
from utils.model_router import route_llm
from utils.logger import get_logger

logger = get_logger(__name__)

SUMMARY_PROMPT = """You are analyzing an Indian health insurance policy document. \
Using ONLY the text below, produce a structured summary. If a field isn't \
present in the text, write "Not specified in document" — never invent details.

Respond using exactly these labeled sections:
Policy Name:
Purpose:
Coverage:
Eligibility:
Benefits:
Waiting Period:
Claim Process:
Premium:
Exclusions:
Important Notes:

DOCUMENT TEXT:
{document_text}
"""

CLAUSE_PROMPT = """Extract the following clauses from this insurance policy text. \
Use ONLY the text provided — if a clause isn't present, write "Not found". \
Never fabricate details.

Respond using exactly these labeled sections:
Eligibility:
Coverage:
Benefits:
Premium:
Waiting Period:
Claim Process:
Hospital Network:
Exclusions:
Important Conditions:

DOCUMENT TEXT:
{document_text}
"""


def _parse_labeled_sections(text: str, labels: List[str]) -> Dict[str, str]:
    sections = {}
    for i, label in enumerate(labels):
        marker = f"{label}:"
        if marker not in text:
            continue
        start = text.index(marker) + len(marker)
        end = len(text)
        for next_label in labels[i + 1:]:
            next_marker = f"{next_label}:"
            if next_marker in text[start:]:
                end = start + text[start:].index(next_marker)
                break
        sections[label] = text[start:end].strip()
    return sections


def generate_policy_summary(document_text: str, provider: str = "OpenAI") -> Dict[str, Any]:
    """One LLM call, structured into labeled fields. Truncates very long
    documents to a safe context size rather than failing."""
    truncated = document_text[:12000]
    prompt = SUMMARY_PROMPT.format(document_text=truncated)
    result = route_llm(provider=provider, prompt=prompt, context=truncated)
    labels = ["Policy Name", "Purpose", "Coverage", "Eligibility", "Benefits",
              "Waiting Period", "Claim Process", "Premium", "Exclusions", "Important Notes"]
    sections = _parse_labeled_sections(result["content"], labels)
    logger.info("Generated policy summary via %s", provider)
    return {"sections": sections, "raw": result["content"], "provider": provider, "model": result["model"]}


def extract_clauses(document_text: str, provider: str = "OpenAI") -> Dict[str, Any]:
    """One LLM call, extracting clause categories into labeled fields."""
    truncated = document_text[:12000]
    prompt = CLAUSE_PROMPT.format(document_text=truncated)
    result = route_llm(provider=provider, prompt=prompt, context=truncated)
    labels = ["Eligibility", "Coverage", "Benefits", "Premium", "Waiting Period",
              "Claim Process", "Hospital Network", "Exclusions", "Important Conditions"]
    sections = _parse_labeled_sections(result["content"], labels)
    logger.info("Extracted clauses via %s", provider)
    return {"sections": sections, "raw": result["content"], "provider": provider, "model": result["model"]}
