"""
tracing.py
Optional LangSmith tracing for the RAG pipeline. If LANGCHAIN_API_KEY /
LANGSMITH_API_KEY is not set, `traceable` becomes a transparent no-op so the
app runs identically without observability configured.
"""

import os
from dotenv import load_dotenv
load_dotenv()
LANGSMITH_ENABLED = bool(
    os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
)

if LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "lang concepts")
    try:
        from langsmith import traceable as _traceable
    except ImportError:
        LANGSMITH_ENABLED = False
        _traceable = None
else:
    _traceable = None


def traceable(name: str = None, run_type: str = "chain"):
    """Decorator that traces to LangSmith when enabled, otherwise passes through."""
    if LANGSMITH_ENABLED and _traceable is not None:
        return _traceable(name=name, run_type=run_type)

    def _noop_decorator(func):
        return func

    return _noop_decorator


def is_tracing_enabled() -> bool:
    return LANGSMITH_ENABLED
