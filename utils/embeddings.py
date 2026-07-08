"""
utils/embeddings.py
Phase 6 — local embedding model initialization, caching, and error handling.

Previously this module wrapped OpenAIEmbeddings, which requires OpenAI API
credits and can fail with `openai.RateLimitError` / `insufficient_quota` when
the account has no embedding quota. It now wraps a local HuggingFace
Sentence-Transformer model (`sentence-transformers/all-MiniLM-L6-v2` by
default) via `langchain_huggingface.HuggingFaceEmbeddings`, so embeddings run
100% locally and never call the OpenAI API.

Nothing else changes: the LLM providers (OpenAI / Groq / NVIDIA NIM) are
routed separately in utils/model_router.py and are completely unaffected —
only *embeddings* moved off OpenAI.

Responsibilities kept in this single module, per its original design intent
("wraps ... embedding model instantiation so it stays swappable"):
  - initialize the embedding model
  - cache it (loaded once per process, not once per query/build)
  - return the embedding object
  - fail cleanly with a clear, user-facing message if the model can't load
"""

from typing import Optional

from config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# Kept as a module-level constant (in addition to config.py) so existing code
# that imports EMBEDDING_MODEL_NAME directly from this module keeps working.
EMBEDDING_MODEL_NAME = CONFIG.embedding.model_name

# Module-level cache: the model is loaded once per process and reused for
# every subsequent "Build KB" click, KB load, or query — it is NOT reloaded
# on every call.
_embedding_model = None


class EmbeddingModelUnavailableError(RuntimeError):
    """
    Raised when the local HuggingFace embedding model cannot be loaded
    (missing dependency, no internet for the first-time model download,
    corrupted cache, etc.). Callers (app.py) catch this specifically to show
    a clean Streamlit message instead of a raw stack trace/crash.
    """


def get_embedding_model(model_name: Optional[str] = None):
    """
    Return a cached local HuggingFace embeddings instance.

    The model is downloaded from the HuggingFace Hub on first use (requires
    internet access once) and then cached on disk by HuggingFace itself for
    offline reuse afterward. This function additionally caches the
    instantiated object in-process so it is created only once, not on every
    'Build KB' click or every query.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    resolved_model_name = model_name or EMBEDDING_MODEL_NAME

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        logger.error("langchain-huggingface is not installed: %s", exc)
        raise EmbeddingModelUnavailableError(
            "Unable to load embedding model.\n\n"
            "The 'langchain-huggingface' package is not installed. "
            "Run `pip install -r requirements.txt` and try again."
        ) from exc

    try:
        logger.info("Loading local HuggingFace embedding model: %s", resolved_model_name)
        _embedding_model = HuggingFaceEmbeddings(model_name=resolved_model_name)
        logger.info("Embedding model '%s' loaded and cached for this session.", resolved_model_name)
    except Exception as exc:  # noqa: BLE001 — any load failure must fail cleanly, not crash the app
        logger.error("Failed to load embedding model '%s': %s", resolved_model_name, exc)
        raise EmbeddingModelUnavailableError(
            "Unable to load embedding model.\n\n"
            "Please check your internet connection for the first-time model "
            "download, then try building the knowledge base again."
        ) from exc

    return _embedding_model
