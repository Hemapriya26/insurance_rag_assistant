"""
vectorstore.py
Builds, persists, and loads the FAISS vector index for retrieval.

Phase 6: embeddings now come from utils.embeddings.get_embedding_model(),
which uses a local HuggingFace Sentence-Transformer model instead of
OpenAIEmbeddings. This module's FAISS logic is otherwise unchanged.

Phase 6.1: load_vectorstore() no longer lets a raw FAISS/pickle error crash
the app on startup. A previously persisted index built with a different
embedding model (different vector dimension) — or one whose files are
truncated/corrupted — now raises a clean IncompatibleIndexError instead of
an unhandled low-level exception, so app.py can catch it and keep the UI
usable (prompting a KB rebuild) rather than crashing before Streamlit even
renders a page.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.embeddings import get_embedding_model, EmbeddingModelUnavailableError
from utils.logger import get_logger

logger = get_logger(__name__)

INDEX_DIR = "data/vectorstore"


class IncompatibleIndexError(RuntimeError):
    """
    Raised when a previously persisted FAISS index at INDEX_DIR cannot be
    loaded with the current embedding model — typically because it was built
    with a different embedding model/dimension (e.g. an older OpenAI-based
    index), or because the index files are corrupted/truncated. The fix is
    always the same: clear the knowledge base and rebuild it.
    """


def build_vectorstore(chunks: List[Document], persist_dir: str = INDEX_DIR) -> FAISS:
    """Embed chunks and build a FAISS index, then persist it to disk."""
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    return vectorstore


def load_vectorstore(persist_dir: str = INDEX_DIR) -> Optional[FAISS]:
    """
    Load a previously persisted FAISS index, if one exists.

    Any failure while reading the on-disk index (dimension mismatch from an
    older embedding model, truncated/corrupted files, etc.) is caught here
    and re-raised as IncompatibleIndexError with a clear, actionable message,
    instead of letting the underlying FAISS/pickle exception propagate
    uncaught. EmbeddingModelUnavailableError (a genuinely different problem —
    the embedding model itself couldn't load) is left to propagate as-is so
    callers can distinguish the two cases.
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return None

    embeddings = get_embedding_model()  # may raise EmbeddingModelUnavailableError

    try:
        return FAISS.load_local(
            persist_dir, embeddings, allow_dangerous_deserialization=True
        )
    except EmbeddingModelUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 — any FAISS/pickle read failure
        logger.warning(
            "Could not load existing FAISS index at '%s' — likely built with a "
            "different embedding model or corrupted (%s).", persist_dir, exc,
        )
        raise IncompatibleIndexError(
            "The existing knowledge base index could not be loaded. This "
            "usually happens when it was built with a different embedding "
            "model than the one currently configured, or the index files are "
            "corrupted.\n\n"
            "Click '🗑️ Clear KB' in the sidebar, then re-upload your PDFs and "
            "click '⚙️ Build KB' to rebuild it."
        ) from exc


def similarity_search_with_score(vectorstore: FAISS, query: str, k: int = 4):
    """Return (Document, score) tuples for the top-k most similar chunks."""
    return vectorstore.similarity_search_with_score(query, k=k)
