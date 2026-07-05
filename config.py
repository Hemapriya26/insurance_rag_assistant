"""
config.py
Phase 3 — central configuration for the Insurance RAG Assistant.
Consolidates settings that were previously scattered as magic numbers/strings
across modules. Existing modules are unaffected unless they explicitly import
from here.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_initial: int = 10          # candidates pulled from FAISS + BM25 each
    top_k_final: int = 4             # chunks passed to the LLM after reranking
    rrf_k: int = 60                  # Reciprocal Rank Fusion constant
    chunk_size: int = 1000
    chunk_overlap: int = 150


@dataclass(frozen=True)
class SecurityConfig:
    max_file_size_mb: int = 25
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    max_files_per_upload: int = 15


@dataclass(frozen=True)
class MemoryConfig:
    max_turns: int = 8               # turns kept verbatim before summarization
    summarize_after_turns: int = 12


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Insurance Policy RAG Assistant"
    version: str = "3.0.0"
    default_provider: str = "OpenAI"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


CONFIG = AppConfig()
