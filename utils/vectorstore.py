"""
vectorstore.py
Builds, persists, and loads the FAISS vector index for retrieval.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.embeddings import get_embedding_model

INDEX_DIR = "data/vectorstore"


def build_vectorstore(chunks: List[Document], persist_dir: str = INDEX_DIR) -> FAISS:
    """Embed chunks and build a FAISS index, then persist it to disk."""
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    return vectorstore


def load_vectorstore(persist_dir: str = INDEX_DIR) -> Optional[FAISS]:
    """Load a previously persisted FAISS index, if one exists.

    Checks for the actual FAISS index files rather than just a non-empty
    directory, since the directory legitimately contains only a .gitkeep
    placeholder on a fresh deploy (before anyone has built a knowledge base).
    """
    index_file = os.path.join(persist_dir, "index.faiss")
    pkl_file = os.path.join(persist_dir, "index.pkl")
    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        return None
    embeddings = get_embedding_model()
    return FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )


def similarity_search_with_score(vectorstore: FAISS, query: str, k: int = 4):
    """Return (Document, score) tuples for the top-k most similar chunks."""
    return vectorstore.similarity_search_with_score(query, k=k)
