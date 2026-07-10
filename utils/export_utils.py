"""
export_utils.py
Phase 2.5 — export the current chat session as Markdown or PDF.
No backend/RAG logic touched; operates purely on st.session_state.messages.
"""

from typing import List, Dict, Any


def export_as_markdown(messages: List[Dict[str, Any]]) -> str:
    lines = ["# Insurance Policy RAG Assistant — Conversation Export\n"]
    for msg in messages:
        role = "🧑 You" if msg["role"] == "user" else "🛡️ Assistant"
        lines.append(f"### {role}\n{msg['content']}\n")
        if msg["role"] == "assistant" and "confidence" in msg:
            lines.append(
                f"*Confidence: {msg['confidence']} · Model: {msg.get('provider','')} "
                f"({msg.get('model','')})*\n"
            )
    return "\n".join(lines)


def export_as_pdf(messages: List[Dict[str, Any]]) -> bytes:
    """Render the conversation to a simple PDF. Requires fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(pdf.l_margin, pdf.t_margin)
    pdf.cell(0, 10, "Insurance Policy RAG Assistant - Conversation")
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(12)

    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, role)
        pdf.set_x(pdf.l_margin)  # fpdf2's multi_cell leaves the cursor at the
                                  # right edge by default; reset before the
                                  # next multi_cell or it has zero width to work with.
        pdf.set_font("Helvetica", "", 10)
        safe_text = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_text)
        if msg["role"] == "assistant" and "confidence" in msg:
            meta = f"Confidence: {msg['confidence']} | Model: {msg.get('provider','')}"
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 6, meta)
        pdf.ln(3)

    # fpdf2's output(name=...) treats a non-empty `name` as a filesystem path
    # and opens it with open(name, 'wb') — passing a BytesIO there fails
    # (works accidentally on some local fpdf2 versions, but not on Streamlit
    # Cloud's). Calling output() with no arguments returns the PDF content
    # directly as a bytearray, with no disk write involved.
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)


def export_as_json(messages: List[Dict[str, Any]]) -> str:
    """
    Phase 4 addition — export the conversation as JSON including full
    metadata (confidence, provider, model, timing, intent, groundedness)
    for each assistant message, not just the display text.
    """
    import json
    from datetime import datetime

    payload = {
        "export_type": "insurance-rag-assistant-conversation",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "message_count": len(messages),
        "messages": messages,
    }
    return json.dumps(payload, indent=2, default=str)
