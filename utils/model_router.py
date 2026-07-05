"""
model_router.py
Pluggable LLM provider routing for Phase 2: OpenAI, Groq, NVIDIA NIM.
All providers share the same system prompt contract defined in rag_chain.py.
"""

import os
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

PROVIDER_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Groq": "groq/compound-mini",
    "NVIDIA NIM": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}


def _get_client(provider: str):
    """Instantiate the correct chat client for the requested provider."""
    if provider == "OpenAI":
        return ChatOpenAI(model=PROVIDER_MODELS["OpenAI"], temperature=0)

    if provider == "Groq":
        if not GROQ_AVAILABLE:
            raise RuntimeError("langchain-groq not installed. Add it to requirements.txt.")
        return ChatGroq(model=PROVIDER_MODELS["Groq"], temperature=0)

    if provider == "NVIDIA NIM":
        # NIM exposes an OpenAI-compatible endpoint
        return ChatOpenAI(
            model=PROVIDER_MODELS["NVIDIA NIM"],
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0,
        )

    raise ValueError(f"Unknown provider: {provider}")


def route_llm(provider: str, prompt: str, context: str) -> Dict[str, Any]:
    """
    Send a fully-formatted prompt (system prompt + context + question already
    merged by rag_chain.py) to the selected provider and return the response
    plus latency metadata for observability.
    """
    start = time.time()
    llm = _get_client(provider)
    response = llm.invoke(prompt)
    latency = round(time.time() - start, 2)

    return {
        "content": response.content,
        "model": PROVIDER_MODELS.get(provider, provider),
        "provider": provider,
        "latency_seconds": latency,
    }
print("Loaded model_router.py")
print(PROVIDER_MODELS)


# ---------------------------------------------------------------------------
# Phase 4 addition — streaming variant. Appended only; nothing above this
# line was changed. Reuses _get_client so both paths share provider setup.
# LangChain's BaseChatModel.stream() is uniform across OpenAI/Groq/NIM: if a
# provider doesn't support true server-side token streaming, it yields the
# full response as a single chunk instead of failing.
# ---------------------------------------------------------------------------
def stream_llm(provider: str, prompt: str):
    """
    Yield text deltas from the selected provider as they arrive. Falls back
    to yielding the full response in one chunk if the provider/model doesn't
    support incremental streaming (still correct, just not incremental).
    """
    llm = _get_client(provider)
    try:
        for chunk in llm.stream(prompt):
            delta = getattr(chunk, "content", "") or ""
            if delta:
                yield delta
    except Exception:
        # Graceful fallback: no incremental streaming available for this
        # provider/model — return the complete answer as a single chunk.
        response = llm.invoke(prompt)
        yield response.content