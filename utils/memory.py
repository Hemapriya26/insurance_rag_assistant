"""
utils/memory.py
Phase 3, Module 1 — Conversational Memory.
Keeps recent turns verbatim, summarizes older turns once the conversation
grows past a threshold, and produces a compact "memory context" string that
rag_chain.py prepends to the retrieval query so follow-up questions
("what about the waiting period for that?") resolve correctly.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    turns: List[Turn] = field(default_factory=list)
    summary: str = ""

    def add_turn(self, question: str, answer: str):
        self.turns.append(Turn(question, answer))
        if len(self.turns) > CONFIG.memory.summarize_after_turns:
            self._summarize_oldest()

    def _summarize_oldest(self):
        """Token-aware trim: collapse everything but the most recent max_turns
        into a short running summary (heuristic, no extra LLM call needed)."""
        keep = CONFIG.memory.max_turns
        to_summarize = self.turns[:-keep]
        self.turns = self.turns[-keep:]
        condensed = "; ".join(f"Q: {t.question[:80]}" for t in to_summarize)
        self.summary = (self.summary + " " + condensed).strip()[-1200:]
        logger.info("Conversation memory summarized (%d turns condensed)", len(to_summarize))

    def context_string(self) -> str:
        """Compact text block for grounding follow-up questions."""
        parts = []
        if self.summary:
            parts.append(f"Earlier conversation summary: {self.summary}")
        for t in self.turns[-CONFIG.memory.max_turns:]:
            parts.append(f"Previous Q: {t.question}\nPrevious A: {t.answer[:300]}")
        return "\n".join(parts)

    def reset(self):
        self.turns = []
        self.summary = ""
        logger.info("Conversation memory reset")


def get_or_create_memory(session_state) -> ConversationMemory:
    if "conversation_memory" not in session_state:
        session_state.conversation_memory = ConversationMemory()
    return session_state.conversation_memory
