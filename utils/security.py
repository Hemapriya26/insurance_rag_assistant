"""
utils/security.py
Phase 3, Module 13 — Security.
Upload validation (type/size/count), input sanitization for chat queries,
and safe error formatting that never leaks stack traces or API keys to the UI.
"""

import os
import re
from typing import List, Tuple
from config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_uploads(uploaded_files) -> Tuple[bool, List[str]]:
    """Return (is_valid, list_of_error_messages). Does not raise."""
    errors = []

    if len(uploaded_files) > CONFIG.security.max_files_per_upload:
        errors.append(f"Too many files: max {CONFIG.security.max_files_per_upload} per upload.")

    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in CONFIG.security.allowed_extensions:
            errors.append(f"'{f.name}': unsupported file type ({ext}). Only PDF is allowed.")
        size_mb = f.size / (1024 * 1024)
        if size_mb > CONFIG.security.max_file_size_mb:
            errors.append(f"'{f.name}': {size_mb:.1f}MB exceeds the {CONFIG.security.max_file_size_mb}MB limit.")

    return (len(errors) == 0), errors


def sanitize_query(text: str, max_length: int = 2000) -> str:
    """Strip control characters and cap length; does not alter legitimate content."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned.strip()[:max_length]


def mask_key(key: str) -> str:
    if not key:
        return "not set"
    return f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "set"


def safe_error_message(exc: Exception) -> str:
    """User-facing error message that never exposes internals or keys."""
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return "Something went wrong processing your request. Please try again or check the logs."
