"""
utils/document_intelligence.py
Phase 3, Module 10 — Document Intelligence.
Document-level statistics, search-within-corpus, and per-source summaries
built entirely from already-chunked Documents (no re-parsing needed).
"""

from typing import List, Dict, Any
from collections import defaultdict
import hashlib
import os
from langchain_core.documents import Document


def document_statistics(chunks: List[Document]) -> Dict[str, Dict[str, Any]]:
    stats = defaultdict(lambda: {"chunks": 0, "characters": 0, "pages": set()})
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        stats[src]["chunks"] += 1
        stats[src]["characters"] += len(chunk.page_content)
        if "page" in chunk.metadata:
            stats[src]["pages"].add(chunk.metadata["page"])

    return {
        src: {"chunks": d["chunks"], "characters": d["characters"], "pages": len(d["pages"]) or "n/a"}
        for src, d in stats.items()
    }


def search_within_documents(chunks: List[Document], term: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not term:
        return []
    term_lower = term.lower()
    hits = []
    for chunk in chunks:
        if term_lower in chunk.page_content.lower():
            idx = chunk.page_content.lower().index(term_lower)
            snippet = chunk.page_content[max(0, idx - 60): idx + 60]
            hits.append({
                "source": chunk.metadata.get("source"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "snippet": f"...{snippet}...",
            })
            if len(hits) >= limit:
                break
    return hits


# ---------------------------------------------------------------------------
# Phase 5 additions — new functions only, nothing above this line was changed.
# ---------------------------------------------------------------------------


def detect_duplicate_chunks(chunks: List[Document]) -> List[Dict[str, Any]]:
    """Detect chunks with identical content (exact-match hash), which usually
    indicates a PDF was uploaded twice or a page repeats boilerplate text."""
    seen: Dict[str, Dict[str, Any]] = {}
    duplicates = []
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
        if content_hash in seen:
            duplicates.append({
                "source": chunk.metadata.get("source"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "duplicate_of_source": seen[content_hash]["source"],
                "duplicate_of_chunk_index": seen[content_hash]["chunk_index"],
            })
        else:
            seen[content_hash] = {"source": chunk.metadata.get("source"), "chunk_index": chunk.metadata.get("chunk_index")}
    return duplicates


def detect_duplicate_files(upload_dir: str) -> List[Dict[str, Any]]:
    """Detect duplicate PDF files in the uploads folder by content hash."""
    if not os.path.isdir(upload_dir):
        return []
    seen: Dict[str, str] = {}
    duplicates = []
    for filename in os.listdir(upload_dir):
        path = os.path.join(upload_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in seen:
            duplicates.append({"file": filename, "duplicate_of": seen[file_hash]})
        else:
            seen[file_hash] = filename
    return duplicates


def knowledge_base_health_score(
    num_chunks: int, has_bm25: bool, avg_chunk_chars: float, num_duplicates: int,
) -> Dict[str, Any]:
    """
    Heuristic 0-100 health score for the knowledge base — NOT a certified
    quality metric, just a quick directional signal combining coverage
    (chunk count), retrieval mode (hybrid vs FAISS-only), chunk size
    sanity, and duplicate penalty.
    """
    score = 0
    reasons = []

    if num_chunks == 0:
        return {"score": 0, "label": "Not Built", "reasons": ["No knowledge base built yet."]}

    score += min(40, num_chunks)  # up to 40 pts for having reasonable chunk coverage
    reasons.append(f"+{min(40, num_chunks)} for {num_chunks} chunks")

    if has_bm25:
        score += 25
        reasons.append("+25 for hybrid retrieval (FAISS + BM25) active")
    else:
        reasons.append("+0 — hybrid retrieval not active (BM25 index missing)")

    if 200 <= avg_chunk_chars <= 2000:
        score += 25
        reasons.append("+25 for healthy average chunk size")
    else:
        score += 10
        reasons.append("+10 — chunk size outside the typical healthy range (200-2000 chars)")

    penalty = min(10, num_duplicates * 2)
    score -= penalty
    if penalty:
        reasons.append(f"-{penalty} for {num_duplicates} duplicate chunk(s) detected")

    score = max(0, min(100, score))
    label = "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
    return {"score": score, "label": label, "reasons": reasons}
