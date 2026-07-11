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

    # Handle both PDF libraries that may be resolved as `fpdf` at import time:
    #   - legacy PyFPDF 1.7.2: output() with no args returns a plain `str`
    #     (Latin-1 encoded text), which must be .encode('latin-1') to get bytes.
    #   - fpdf2 (modern, maintained fork): output() with no args returns a
    #     `bytearray` directly — no disk write, no encoding needed.
    # Previously this assumed fpdf2 unconditionally and called bytes() on a
    # str, which raises "TypeError: string argument without an encoding" on
    # environments that resolved the legacy `fpdf` package instead.
    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return bytes(raw)


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
