"""
utils/document_search.py
Phase 5 — Document Search (Document Intelligence page).
Thin wrapper around utils.document_intelligence.search_within_documents()
that adds HTML highlighting of the matched term for display. The underlying
search function is reused unchanged.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from utils.document_intelligence import search_within_documents


def search_with_highlighting(chunks: List[Document], term: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Same results as search_within_documents(), with the matched term
    wrapped in <mark> tags for display (reuses no new CSS — <mark> has a
    sensible browser default)."""
    hits = search_within_documents(chunks, term, limit=limit)
    if not term:
        return hits
    for hit in hits:
        hit["highlighted_snippet"] = hit["snippet"].replace(term, f"<mark>{term}</mark>")
        if term.lower() == term:
            capitalized = term.capitalize()
            hit["highlighted_snippet"] = hit["highlighted_snippet"].replace(capitalized, f"<mark>{capitalized}</mark>")
    return hits
