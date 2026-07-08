"""
utils/document_explorer.py
Phase 5 — Document Intelligence page rendering: Document Explorer, Document
Statistics, Duplicate Detection, and Knowledge Base Health. Combines data
functions from utils.document_intelligence with Streamlit rendering, so
app.py only needs to call one function per section.
"""

import os
from datetime import datetime
from typing import List, Optional
import streamlit as st
from langchain_core.documents import Document

from utils.document_intelligence import (
    document_statistics, detect_duplicate_chunks, detect_duplicate_files,
    knowledge_base_health_score,
)
from utils.embeddings import EMBEDDING_MODEL_NAME


def render_document_explorer(chunks: List[Document], upload_dir: str) -> None:
    """Browse uploaded PDFs: name, page count, chunk count, upload time, expandable chunks."""
    if not chunks:
        st.info("No documents in the knowledge base yet. Build the KB from the sidebar first.")
        return

    stats = document_statistics(chunks)
    by_source: dict = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata.get("source", "unknown"), []).append(chunk)

    for source, source_chunks in by_source.items():
        file_path = os.path.join(upload_dir, source)
        upload_time = "unknown"
        if os.path.exists(file_path):
            upload_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")

        doc_stats = stats.get(source, {})
        with st.expander(f"📄 {source} — {doc_stats.get('chunks', 0)} chunks, {doc_stats.get('pages', 'n/a')} pages"):
            st.markdown(
                f"**Uploaded:** {upload_time}  \n"
                f"**Chunks:** {doc_stats.get('chunks', 0)}  \n"
                f"**Characters:** {doc_stats.get('characters', 0):,}  \n"
                f"**Pages:** {doc_stats.get('pages', 'n/a')}"
            )
            for chunk in source_chunks[:50]:
                with st.expander(f"Chunk {chunk.metadata.get('chunk_index', '?')}"):
                    st.caption(chunk.page_content[:800])
                    st.json(chunk.metadata)


def render_document_statistics(chunks: List[Document]) -> None:
    """Total PDFs, pages, chunks, avg chunk length/tokens, largest/smallest doc, total embeddings."""
    if not chunks:
        st.info("No documents to analyze yet.")
        return

    stats = document_statistics(chunks)
    total_pdfs = len(stats)
    total_chunks = sum(s["chunks"] for s in stats.values())
    total_pages = sum(s["pages"] for s in stats.values() if isinstance(s["pages"], int))
    avg_chunk_len = sum(s["characters"] for s in stats.values()) / total_chunks if total_chunks else 0
    avg_tokens = max(1, round(avg_chunk_len / 4))  # ~4 chars/token, standard rough heuristic
    largest = max(stats.items(), key=lambda kv: kv[1]["characters"], default=(None, {}))
    smallest = min(stats.items(), key=lambda kv: kv[1]["characters"], default=(None, {}))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PDFs", total_pdfs)
    c2.metric("Total Pages", total_pages or "n/a")
    c3.metric("Total Chunks", total_chunks)
    c4.metric("Total Embeddings", total_chunks)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Chunk Length", f"{avg_chunk_len:.0f} chars")
    c6.metric("Avg Tokens/Chunk", avg_tokens)
    c7.metric("Largest Document", largest[0] or "n/a")
    c8.metric("Smallest Document", smallest[0] or "n/a")


def render_duplicate_detection(chunks: List[Document], upload_dir: str) -> None:
    """Warn about duplicate PDFs and duplicate chunk content."""
    dup_files = detect_duplicate_files(upload_dir)
    dup_chunks = detect_duplicate_chunks(chunks)

    if not dup_files and not dup_chunks:
        st.success("No duplicates detected.")
        return

    if dup_files:
        st.warning(f"⚠️ {len(dup_files)} duplicate file(s) detected:")
        for d in dup_files:
            st.caption(f"'{d['file']}' is a duplicate of '{d['duplicate_of']}'")

    if dup_chunks:
        st.warning(f"⚠️ {len(dup_chunks)} duplicate chunk(s) detected:")
        for d in dup_chunks[:20]:
            st.caption(
                f"{d['source']} chunk {d['chunk_index']} duplicates "
                f"{d['duplicate_of_source']} chunk {d['duplicate_of_chunk_index']}"
            )


def render_kb_health(vectorstore, bm25_index, chunks: List[Document], index_dir: str) -> None:
    """Embedding model, embedding count, FAISS/BM25 stats, hybrid status, chunk-size sanity, health score."""
    if not chunks or vectorstore is None:
        st.info("Build the knowledge base first to see health metrics.")
        return

    num_chunks = len(chunks)
    avg_chunk_chars = sum(len(c.page_content) for c in chunks) / num_chunks if num_chunks else 0
    dup_chunks = detect_duplicate_chunks(chunks)

    faiss_total = getattr(getattr(vectorstore, "index", None), "ntotal", "n/a")
    index_size_bytes = 0
    if os.path.isdir(index_dir):
        index_size_bytes = sum(
            os.path.getsize(os.path.join(index_dir, f)) for f in os.listdir(index_dir)
            if os.path.isfile(os.path.join(index_dir, f))
        )

    bm25_corpus_size = getattr(bm25_index, "corpus_size", None) if bm25_index else None
    bm25_avgdl = getattr(bm25_index, "avgdl", None) if bm25_index else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Embedding Model", EMBEDDING_MODEL_NAME)
    c2.metric("Embedding Count", num_chunks)
    c3.metric("Index Size on Disk", f"{index_size_bytes / 1024:.1f} KB")

    c4, c5, c6 = st.columns(3)
    c4.metric("FAISS Vectors", faiss_total)
    c5.metric("BM25 Corpus Size", bm25_corpus_size if bm25_corpus_size is not None else "Not built")
    c6.metric("Hybrid Retrieval", "🟢 Active" if bm25_index else "⚪ FAISS-only")

    c7, c8 = st.columns(2)
    c7.metric("Avg Chunk Size", f"{avg_chunk_chars:.0f} chars")
    c8.metric("Avg BM25 Doc Length", f"{bm25_avgdl:.1f} tokens" if bm25_avgdl else "n/a")

    health = knowledge_base_health_score(num_chunks, bool(bm25_index), avg_chunk_chars, len(dup_chunks))
    st.markdown(f"### Knowledge Base Health: {health['score']}/100 — {health['label']}")
    with st.expander("Why this score?"):
        for reason in health["reasons"]:
            st.caption(f"• {reason}")
