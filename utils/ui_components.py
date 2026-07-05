"""
ui_components.py
Reusable Streamlit rendering helpers: chat bubbles, typing indicator,
retrieved-context viewer, and source/confidence formatting.
"""

import streamlit as st

CONFIDENCE_BADGE = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}


def render_chat_message(role: str, content: str):
    """Render a single chat bubble with fade-in animation."""
    with st.chat_message(role):
        st.markdown(f"<div class='glass-card chat-bubble-{role}'>{content}</div>",
                     unsafe_allow_html=True)


def render_typing_indicator():
    """Animated three-dot typing indicator shown while the LLM is generating."""
    st.markdown(
        """
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_meta(confidence: str, provider: str, model: str,
                        retrieval_time: float, generation_time: float):
    """Render confidence badge + model-used + latency metadata row."""
    badge_class = CONFIDENCE_BADGE.get(confidence, "badge-low")
    st.markdown(
        f"""
        <div class="meta-row">
            <span class="badge {badge_class}">Confidence: {confidence}</span>
            <span class="badge badge-model">Model: {provider} ({model})</span>
            <span class="badge badge-time">Retrieval: {retrieval_time}s</span>
            <span class="badge badge-time">Generation: {generation_time}s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources_expander(retrieved_chunks: list, key_prefix: str = "", query: str = None):
    """
    Expandable panel listing retrieved chunks with source + chunk index +
    relevance score. Phase 4: adds a similarity bar and highlights query
    terms within each preview — purely additive; `query` defaults to None so
    existing call sites (which don't pass it) behave exactly as before.
    """
    with st.expander("📚 Sources & Retrieved Context"):
        if not retrieved_chunks:
            st.caption("No chunks retrieved.")
            return
        scores = [c["score"] for c in retrieved_chunks]
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(
                f"**{i}. {chunk['source']} — chunk {chunk['chunk_index']}** "
                f"(relevance score: {chunk['score']})"
            )
            # Similarity indicator: reuses existing .badge-high/.badge-medium/.badge-low
            # classes already defined in style.css — no new CSS added.
            pct = max(4, min(100, round((chunk["score"] / max_score) * 100)))
            bar_class = "badge-high" if pct >= 66 else "badge-medium" if pct >= 33 else "badge-low"
            st.markdown(
                f"<div class='glass-card' style='padding:6px 10px; margin:4px 0 10px 0;'>"
                f"<span class='badge {bar_class}'>{'▓' * (pct // 10)}{'░' * (10 - pct // 10)} {pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            preview = chunk["content"][:500]
            preview = preview + ("..." if len(chunk["content"]) > 500 else "")
            if query:
                for term in {t for t in query.lower().split() if len(t) > 3}:
                    preview = preview.replace(term, f"<mark>{term}</mark>")
                    preview = preview.replace(term.capitalize(), f"<mark>{term.capitalize()}</mark>")
                st.markdown(f"<span style='font-size:0.85rem; opacity:0.85;'>{preview}</span>",
                             unsafe_allow_html=True)
            else:
                st.caption(preview)


def render_header_banner(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="header-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Phase 4 additions — new functions only, nothing above this line was changed.
# ---------------------------------------------------------------------------
def render_timestamp(timestamp: str):
    """Small caption under a chat bubble. Reuses st.caption styling already
    used elsewhere (e.g. chunk previews) — no new CSS."""
    if timestamp:
        st.caption(timestamp)


def render_message_actions(message_index: int, content: str) -> str:
    """
    Copy / Regenerate / Like / Dislike row under an assistant message.
    Returns 'regenerate', 'like', 'dislike', or None if nothing was clicked.
    Uses plain st.button — the same widget type already used throughout the
    sidebar — so it visually matches without introducing new styling.
    """
    action = None
    cols = st.columns([1, 1, 1, 1, 6])
    with cols[0]:
        if st.button("📋", key=f"copy_{message_index}", help="Show copyable text"):
            st.code(content, language=None)
    with cols[1]:
        if st.button("🔁", key=f"regen_{message_index}", help="Regenerate this answer"):
            action = "regenerate"
    with cols[2]:
        if st.button("👍", key=f"like_{message_index}", help="Good answer"):
            action = "like"
    with cols[3]:
        if st.button("👎", key=f"dislike_{message_index}", help="Poor answer"):
            action = "dislike"
    return action


def render_suggested_questions(suggestions: list, key_prefix: str = "sugg") -> str:
    """Clickable follow-up question chips under an assistant response.
    Returns the selected question text, or None. Uses plain st.button in a
    row, matching existing widget style."""
    if not suggestions:
        return None
    selected = None
    st.caption("💡 You might also ask:")
    cols = st.columns(len(suggestions))
    for col, question in zip(cols, suggestions):
        with col:
            if st.button(question, key=f"{key_prefix}_{question[:24]}", use_container_width=True):
                selected = question
    return selected
def render_footer():
    """Application footer."""
    st.markdown(
        """
        <div class="footer">
            <hr>
            <p>
                🛡️ <strong>Insurance Policy RAG Assistant</strong> |
                Built with ❤️ using Streamlit, LangChain, FAISS & OpenAI
            </p>
            <p class="footer-small">
                © 2026 Hemapriya • Enterprise AI Project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )