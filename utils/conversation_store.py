"""
utils/conversation_store.py
Phase 4 — Conversation Management.
Persists named conversations to a local JSON file so history survives app
restarts without adding a database dependency. Each conversation stores its
own message list; the active conversation ID lives in st.session_state.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.json")


def _load_all() -> Dict[str, Any]:
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load conversation store: %s", exc)
        return {}


def _save_all(conversations: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
    except OSError as exc:
        logger.warning("Could not save conversation store: %s", exc)


def create_conversation(title: str = "New Conversation") -> str:
    conversations = _load_all()
    conv_id = str(uuid.uuid4())[:8]
    conversations[conv_id] = {
        "title": title, "messages": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_all(conversations)
    logger.info("Created conversation %s (%s)", conv_id, title)
    return conv_id


def save_messages(conv_id: str, messages: List[Dict[str, Any]]) -> None:
    conversations = _load_all()
    if conv_id not in conversations:
        conversations[conv_id] = {"title": "Conversation", "messages": [], "created_at": datetime.now().isoformat(timespec="seconds")}
    conversations[conv_id]["messages"] = messages
    _save_all(conversations)


def rename_conversation(conv_id: str, new_title: str) -> None:
    conversations = _load_all()
    if conv_id in conversations:
        conversations[conv_id]["title"] = new_title
        _save_all(conversations)
        logger.info("Renamed conversation %s to '%s'", conv_id, new_title)


def delete_conversation(conv_id: str) -> None:
    conversations = _load_all()
    if conv_id in conversations:
        del conversations[conv_id]
        _save_all(conversations)
        logger.info("Deleted conversation %s", conv_id)


def list_conversations() -> Dict[str, Any]:
    return _load_all()


def search_conversations(term: str) -> Dict[str, Any]:
    """Search conversation titles and message content for `term` (case-insensitive)."""
    if not term:
        return {}
    term_lower = term.lower()
    conversations = _load_all()
    matches = {}
    for conv_id, data in conversations.items():
        title_match = term_lower in data.get("title", "").lower()
        content_match = any(term_lower in m.get("content", "").lower() for m in data.get("messages", []))
        if title_match or content_match:
            matches[conv_id] = data
    return matches


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    return _load_all().get(conv_id)
