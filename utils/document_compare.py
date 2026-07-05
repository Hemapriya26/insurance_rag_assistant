"""
utils/document_compare.py
Phase 5 — Policy Comparison (Document Intelligence).
One LLM call comparing two documents' text, producing a markdown table.
User-triggered from the Document Intelligence page.
"""

from typing import Dict, Any
from utils.model_router import route_llm
from utils.logger import get_logger

logger = get_logger(__name__)

COMPARE_PROMPT = """Compare these two Indian health insurance policy documents \
using ONLY the text provided. Never invent details not present in the text — \
write "Not specified" where information is missing from either document.

Produce a markdown table with these rows: Eligibility, Coverage, Benefits,
Premium, Waiting Period, Claim Process, Exclusions. Columns: the row label,
"Document A", "Document B". After the table, add a one-paragraph plain-text
summary of the key differences.

DOCUMENT A ({name_a}):
{text_a}

DOCUMENT B ({name_b}):
{text_b}
"""


def compare_policies(
    name_a: str, text_a: str, name_b: str, text_b: str, provider: str = "OpenAI",
) -> Dict[str, Any]:
    """One LLM call producing a comparison table + summary. Each document's
    text is truncated to a safe size so the combined prompt stays bounded."""
    prompt = COMPARE_PROMPT.format(
        name_a=name_a, text_a=text_a[:6000], name_b=name_b, text_b=text_b[:6000],
    )
    result = route_llm(provider=provider, prompt=prompt, context=text_a[:6000] + text_b[:6000])
    logger.info("Compared policies '%s' vs '%s' via %s", name_a, name_b, provider)
    return {"comparison_markdown": result["content"], "provider": provider, "model": result["model"]}
