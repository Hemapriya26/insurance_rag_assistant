"""
embeddings.py
Wraps OpenAI embedding model instantiation so it stays swappable in later phases
(e.g. Phase 2 may add Groq / NIM embeddings).
"""

from langchain_openai import OpenAIEmbeddings


def get_embedding_model(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Return a configured OpenAI embeddings instance."""
    return OpenAIEmbeddings(model=model_name)
