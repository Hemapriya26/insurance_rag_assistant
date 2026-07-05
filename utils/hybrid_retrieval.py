"""
utils/hybrid_retrieval.py
Phase 3, Module 2 — Hybrid Retrieval.
Combines the existing FAISS semantic search (unchanged, from vectorstore.py)
with a BM25 keyword index using Reciprocal Rank Fusion. FAISS remains fully
functional standalone; this module only adds a second signal on top.

Also persists the raw chunk list to disk alongside the FAISS index so BM25
can be rebuilt automatically on app restart — without this, a previously
built knowledge base would load its FAISS index on startup but silently
lose the BM25 half of hybrid retrieval until "Build KB" was clicked again.
"""

import os
import pickle
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config import CONFIG
from utils.vectorstore import similarity_search_with_score
from utils.logger import get_logger

logger = get_logger(__name__)

CHUNKS_CACHE_FILENAME = "chunks.pkl"


def build_bm25_index(chunks: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
    """Build a BM25 index over the same chunks used for the FAISS index."""
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index built over %d chunks", len(chunks))
    return bm25, chunks


def save_chunks_to_disk(chunks: List[Document], persist_dir: str) -> None:
    """Persist raw chunks alongside the FAISS index so BM25 can be rebuilt
    automatically after a restart, without needing to re-run 'Build KB'."""
    try:
        os.makedirs(persist_dir, exist_ok=True)
        path = os.path.join(persist_dir, CHUNKS_CACHE_FILENAME)
        with open(path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info("Persisted %d chunks for BM25 restart-recovery to %s", len(chunks), path)
    except OSError as exc:
        logger.warning("Could not persist chunks for BM25 recovery: %s", exc)


def load_chunks_from_disk(persist_dir: str) -> Optional[List[Document]]:
    """Load previously persisted chunks, if present, to rebuild BM25 on startup."""
    path = os.path.join(persist_dir, CHUNKS_CACHE_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Restored %d chunks from disk for BM25 hybrid retrieval", len(chunks))
        return chunks
    except (OSError, pickle.PickleError) as exc:
        logger.warning("Could not load persisted chunks for BM25 recovery: %s", exc)
        return None


def _bm25_search(bm25: BM25Okapi, corpus: List[Document], query: str, k: int) -> List[Document]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus[i] for i in ranked]


def hybrid_search(
    vectorstore, bm25_index, bm25_corpus: List[Document], query: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Run FAISS semantic search and BM25 keyword search in parallel, fuse the
    two ranked lists with Reciprocal Rank Fusion (RRF), and return documents
    with their fused RRF score (used purely for display/reranking input).
    """
    k = k or CONFIG.retrieval.top_k_initial
    rrf_k = CONFIG.retrieval.rrf_k

    semantic_results = similarity_search_with_score(vectorstore, query, k=k)
    semantic_docs = [doc for doc, _ in semantic_results]

    keyword_docs = []
    if bm25_index is not None and bm25_corpus:
        keyword_docs = _bm25_search(bm25_index, bm25_corpus, query, k)

    def doc_key(doc: Document) -> str:
        return f"{doc.metadata.get('source')}::{doc.metadata.get('chunk_index')}"

    rrf_scores = {}
    doc_lookup = {}

    for rank, doc in enumerate(semantic_docs):
        key = doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank + 1)
        doc_lookup[key] = doc

    for rank, doc in enumerate(keyword_docs):
        key = doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank + 1)
        doc_lookup[key] = doc

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_lookup[key], round(score, 5)) for key, score in fused[:k]]
