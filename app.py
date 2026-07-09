"""
app.py
Insurance Policy RAG Assistant — Phase 4 (User Experience Upgrade)
Existing Phase 2/3 layout, sidebar order, CSS, and widgets are preserved.
New elements (chat action buttons, conversation manager, analytics/export
expanders, streaming) are added additively, reusing existing CSS classes
(glass-card, badge-*) so they blend into the current visual language rather
than introducing a new design.

Streaming uses Streamlit's documented fragment + background-thread pattern
(requires Streamlit >= 1.33 for st.fragment; this project was verified
against 1.58). The background thread only pushes text onto a queue — it
never touches st.session_state or calls any st.* function directly, which
is the safe boundary for combining threads with Streamlit.
"""

import os
import time
import queue as queue_module
import threading
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from utils.loader import save_uploaded_files, load_pdfs
from utils.chunker import chunk_documents
from utils.vectorstore import build_vectorstore, load_vectorstore, IncompatibleIndexError, INDEX_DIR
from utils.rag_chain import answer_question, prepare_retrieval, finalize_answer
from utils.hybrid_retrieval import build_bm25_index, save_chunks_to_disk, load_chunks_from_disk
from utils.reranker import reranker_mode
from utils.tracing import is_tracing_enabled
from utils.model_router import stream_llm, PROVIDER_MODELS
from utils.memory import get_or_create_memory
from utils.security import validate_uploads, sanitize_query, safe_error_message
from utils.embeddings import EMBEDDING_MODEL_NAME, EmbeddingModelUnavailableError
from utils.evaluation import evaluate_response
from utils.export_utils import export_as_markdown, export_as_pdf, export_as_json
from utils.suggestions import generate_suggestions
from utils import conversation_store
from utils.ui_components import (
    render_chat_message, render_typing_indicator, render_answer_meta,
    render_header_banner, render_timestamp,render_footer, render_sources_expander,
    render_message_actions, render_suggested_questions,
)

load_dotenv()

st.set_page_config(
    page_title="Insurance Policy RAG Assistant",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state (Phase 3 keys unchanged; Phase 4 keys added)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_vectorstore()
    except EmbeddingModelUnavailableError as exc:
        # Don't crash the whole app on startup if the embedding model itself
        # can't load (e.g. first-time model download failed) — fall back to
        # "no KB loaded" and let the user retry from the sidebar.
        st.session_state.vectorstore = None
        st.session_state._embedding_load_error = str(exc)
    except IncompatibleIndexError as exc:
        # A previously persisted index (often built with a different/older
        # embedding model) can't be read. Auto-clear the stale index files so
        # the app doesn't keep failing on every new session/restart — the
        # user just needs to re-upload PDFs and click "Build KB" once.
        import shutil
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
        st.session_state.vectorstore = None
        st.session_state._index_incompatible_warning = str(exc)
    except Exception as exc:  # noqa: BLE001 — last-resort safety net so a
        # startup KB load can never crash the whole app; log it and continue
        # with no KB loaded rather than taking the site down.
        from utils.logger import get_logger as _get_logger
        _get_logger(__name__).error("Unexpected error loading KB at startup: %s", exc, exc_info=True)
        st.session_state.vectorstore = None
        st.session_state._embedding_load_error = (
            "Could not load the existing knowledge base. Please click "
            "'🗑️ Clear KB' in the sidebar, then re-upload your PDFs and rebuild it."
        )
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "query_log" not in st.session_state:
    st.session_state.query_log = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
    if st.session_state.vectorstore is not None:
        from utils.vectorstore import INDEX_DIR as _INDEX_DIR
        _restored_chunks = load_chunks_from_disk(_INDEX_DIR)
        if _restored_chunks:
            st.session_state.all_chunks = _restored_chunks
            st.session_state.bm25_index, _ = build_bm25_index(_restored_chunks)
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# Phase 4 additions
if "conv_id" not in st.session_state:
    st.session_state.conv_id = conversation_store.create_conversation("New Conversation")
if "feedback" not in st.session_state:
    st.session_state.feedback = {}          # {message_index: 'like' | 'dislike'}
if "stream_state" not in st.session_state:
    st.session_state.stream_state = None    # active streaming generation, if any
if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion = None

memory = get_or_create_memory(st.session_state)

# ---------------------------------------------------------------------------
# Theme + CSS injection (UNCHANGED)
# ---------------------------------------------------------------------------
css_path = os.path.join(os.path.dirname(__file__), "utils", "style.css")
with open(css_path) as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
        :root {
            --glass-bg: rgba(30, 30, 46, 0.55);
            --glass-border: rgba(255,255,255,0.08);
            --text-color: #f1f1f6;
            --bg-gradient: linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

render_header_banner(
    "🛡️ Insurance Policy RAG Assistant",
    "Document-grounded answers for CMCHIS · PM-JAY · ESIC · CGHS · Private Insurance",
)

if st.session_state.get("_embedding_load_error"):
    st.error(st.session_state._embedding_load_error)
# Note: an incompatible/stale index (_index_incompatible_warning) is handled
# silently — the stale files are auto-cleared at startup and the event is
# logged to logs/app.log, but no banner is shown to the user. The app simply
# falls back to the normal "no KB loaded yet" welcome screen, exactly as it
# would look on a genuine first run.

# ---------------------------------------------------------------------------
# Background streaming worker — MUST NOT touch st.session_state or st.* calls.
# ---------------------------------------------------------------------------
def _background_stream(provider, prompt, out_queue, stop_event):
    try:
        for delta in stream_llm(provider, prompt):
            if stop_event.is_set():
                break
            out_queue.put(delta)
    except Exception as exc:  # noqa: BLE001 — surface any provider error into the stream itself
        out_queue.put(f"\n\n_(Generation error: {exc})_")
    finally:
        out_queue.put(None)  # sentinel: stream finished


def _start_streaming(query_text, provider, top_k):
    """Kick off retrieval (blocking, fast) then hand generation to a background thread."""
    prepared = prepare_retrieval(
        st.session_state.vectorstore, query_text,
        bm25_index=st.session_state.bm25_index, bm25_corpus=st.session_state.all_chunks,
        memory=memory,
    )
    if not prepared["scored_docs"]:
        result = finalize_answer(query_text, prepared, "", provider, "n/a", 0.0)
        _finish_turn(query_text, result)
        return

    q = queue_module.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_background_stream, args=(provider, prepared["prompt"], q, stop_event), daemon=True,
    )
    st.session_state.stream_state = {
        "active": True, "done": False, "accumulated": "", "queue": q, "stop_event": stop_event,
        "thread": thread, "prepared": prepared, "query": query_text, "provider": provider,
        "start_time": time.time(),
    }
    thread.start()


def _finish_turn(query_text, result):
    """Shared finalization: append message, update memory/log/eval/conversation store."""
    ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "timestamp": ts,
        "sections": result.get("sections", {}), "sources": result["sources"],
        "confidence": result["confidence"], "intent": result.get("intent", "General"),
        "retrieved_chunks": result["retrieved_chunks"], "provider": result["provider"],
        "model": result["model"], "retrieval_time": result["retrieval_time"],
        "generation_time": result["generation_time"], "grounded": result.get("grounded", True),
    })
    memory.add_turn(query_text, result["answer"])
    conversation_store.save_messages(st.session_state.conv_id, st.session_state.messages)
    st.session_state.query_log.append({
        "query": query_text, "answer": result["answer"], "provider": result["provider"],
        "model": result["model"], "retrieval_time": result["retrieval_time"],
        "generation_time": result["generation_time"], "confidence": result["confidence"],
        "intent": result.get("intent", "General"), "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    context_text = "\n".join(c["content"] for c in result["retrieved_chunks"])
    st.session_state.evaluations.append(evaluate_response(query_text, result["answer"], context_text))


@st.fragment(run_every=0.12)
def render_streaming_fragment():
    """Polls the background thread's queue and progressively renders the answer.
    Runs on its own short interval without rerunning the rest of the page —
    this is what lets the Stop button remain responsive mid-generation."""
    state = st.session_state.stream_state
    if not state or not state.get("active"):
        return

    try:
        while True:
            item = state["queue"].get_nowait()
            if item is None:
                state["done"] = True
                break
            state["accumulated"] += item
    except queue_module.Empty:
        pass

    cursor = "▌" if not state["done"] else ""
    st.markdown(
        f"<div class='glass-card chat-bubble-assistant'>{state['accumulated']}{cursor}</div>",
        unsafe_allow_html=True,
    )
    if not state["done"]:
        if st.button("⏹ Stop Generating", key="stop_gen_btn"):
            state["stop_event"].set()
            state["done"] = True

    if state["done"]:
        generation_time = round(time.time() - state["start_time"], 2)
        model_name = PROVIDER_MODELS.get(state["provider"], state["provider"])
        final_text = state["accumulated"].strip() or "Information not found in documents."
        result = finalize_answer(
            state["query"], state["prepared"], final_text,
            state["provider"], model_name, generation_time,
        )
        _finish_turn(state["query"], result)
        st.session_state.stream_state = None
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — existing widgets unchanged, Phase 4 sections added below them
# ---------------------------------------------------------------------------
with st.sidebar:
    toggle_label = "🌙 Switch to Dark Mode" if st.session_state.theme == "light" else "☀️ Switch to Light Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    st.markdown("### 🤖 Model Selection")
    provider = st.selectbox(
        "LLM Provider",
        options=["OpenAI", "Groq", "NVIDIA NIM"],
        help="OpenAI = most accurate · Groq = fastest · NVIDIA NIM = experimental",
    )

    st.markdown("### 📁 Upload Policy PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more insurance PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col_build, col_clear = st.columns(2)
    with col_build:
        build_clicked = st.button("⚙️ Build KB", use_container_width=True, type="primary")
    with col_clear:
        clear_clicked = st.button("🗑️ Clear KB", use_container_width=True)

    if build_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            is_valid, errors = validate_uploads(uploaded_files)
            if not is_valid:
                for e in errors:
                    st.error(e)
            else:
                try:
                    progress = st.progress(0, text="📄 Saving and reading PDFs...")
                    paths = save_uploaded_files(uploaded_files)
                    docs = load_pdfs(paths)
                    progress.progress(25, text="✂️ Chunking documents...")
                    chunks = chunk_documents(docs)
                    progress.progress(50, text="🧠 Creating embeddings & FAISS index...")
                    st.session_state.vectorstore = build_vectorstore(chunks)
                    progress.progress(80, text="🔎 Building keyword (BM25) index...")
                    bm25, corpus = build_bm25_index(chunks)
                    st.session_state.bm25_index = bm25
                    st.session_state.all_chunks = corpus
                    from utils.vectorstore import INDEX_DIR as _INDEX_DIR
                    save_chunks_to_disk(corpus, _INDEX_DIR)
                    progress.progress(100, text="✅ Knowledge base ready")
                    time.sleep(0.3)
                    progress.empty()
                    st.success(f"Knowledge base built from {len(chunks)} chunks across {len(paths)} document(s).")
                    # A fresh, successful build supersedes any earlier
                    # startup warning about a stale/incompatible index — clear
                    # those banners so they don't linger for the rest of the
                    # session after the problem is already resolved.
                    st.session_state.pop("_index_incompatible_warning", None)
                    st.session_state.pop("_embedding_load_error", None)
                except EmbeddingModelUnavailableError as exc:
                    # Clean, specific message for local embedding-model load
                    # failures (e.g. no internet for first-time download) —
                    # shown as-is rather than the generic safe_error_message.
                    progress.empty()
                    st.error(str(exc))
                except Exception as exc:
                    st.error(safe_error_message(exc))

    if clear_clicked:
        from utils.vectorstore import INDEX_DIR
        import shutil
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.all_chunks = []
        st.session_state.bm25_index = None
        st.session_state.query_log = []
        st.session_state.evaluations = []
        st.session_state.feedback = {}
        memory.reset()
        conversation_store.save_messages(st.session_state.conv_id, [])
        st.session_state.pop("_index_incompatible_warning", None)
        st.session_state.pop("_embedding_load_error", None)
        st.success("Knowledge base cleared.")
        st.rerun()

    st.markdown("### 🎛️ Retrieval Settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=2, max_value=10, value=4)

    st.markdown("---")
    kb_status = "🟢 Ready" if st.session_state.vectorstore else "🔴 Not built yet"
    last_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    total_response_time = round(last_msg["retrieval_time"] + last_msg["generation_time"], 2) if last_msg else 0.0
    groundedness_display = "⚪ n/a"
    if last_msg:
        groundedness_display = "🟢 Grounded" if last_msg.get("grounded", True) else "🟡 Review"
    st.markdown(
        f"**Knowledge Base Status:** {kb_status}  \n"
        f"**Embedding Model:** {EMBEDDING_MODEL_NAME}  \n"
        f"**LLM Provider:** {provider}  \n"
        f"**Last Response Time:** {total_response_time}s  \n"
        f"**Conversation Turns:** {len(memory.turns)}  \n"
        f"**Memory Summarized:** {'Yes' if memory.summary else 'No'}  \n"
        f"**LangSmith Tracing:** {'🟢 Enabled' if is_tracing_enabled() else '⚪ Disabled'}  \n"
        f"**Reranker:** {reranker_mode()}  \n"
        f"**Last Groundedness:** {groundedness_display}"
    )

    st.markdown("---")
    with st.expander("💬 Conversations"):
        conv_data = conversation_store.get_conversation(st.session_state.conv_id) or {}
        new_title = st.text_input("Rename this conversation", value=conv_data.get("title", "New Conversation"))
        if new_title and new_title != conv_data.get("title"):
            conversation_store.rename_conversation(st.session_state.conv_id, new_title)

        col_new, col_del = st.columns(2)
        with col_new:
            if st.button("🆕 New", use_container_width=True):
                st.session_state.conv_id = conversation_store.create_conversation("New Conversation")
                st.session_state.messages = []
                memory.reset()
                st.rerun()
        with col_del:
            if st.button("🗑️ Delete", use_container_width=True):
                conversation_store.delete_conversation(st.session_state.conv_id)
                st.session_state.conv_id = conversation_store.create_conversation("New Conversation")
                st.session_state.messages = []
                memory.reset()
                st.rerun()

        all_convs = conversation_store.list_conversations()
        if all_convs:
            options = {cid: d.get("title", cid) for cid, d in all_convs.items()}
            option_ids = list(options.keys())
            current_index = option_ids.index(st.session_state.conv_id) if st.session_state.conv_id in option_ids else 0
            selected = st.selectbox("Switch conversation", options=option_ids,
                                     format_func=lambda cid: options[cid], index=current_index)
            if selected != st.session_state.conv_id:
                st.session_state.conv_id = selected
                st.session_state.messages = conversation_store.get_conversation(selected).get("messages", [])
                st.rerun()

        search_term = st.text_input("🔎 Search all conversations", key="conv_search")
        if search_term:
            matches = conversation_store.search_conversations(search_term)
            for cid, data in matches.items():
                st.caption(f"• {data.get('title', cid)} ({len(data.get('messages', []))} messages)")

    with st.expander("💾 Export Conversation"):
        if st.session_state.messages:
            st.download_button("⬇️ Markdown", export_as_markdown(st.session_state.messages),
                                "conversation.md", "text/markdown", use_container_width=True)
            st.download_button("⬇️ JSON (with metadata)", export_as_json(st.session_state.messages),
                                "conversation.json", "application/json", use_container_width=True)
            try:
                st.download_button("⬇️ PDF", export_as_pdf(st.session_state.messages),
                                    "conversation.pdf", "application/pdf", use_container_width=True)
            except ImportError:
                st.caption("Install `fpdf`/`fpdf2` to enable PDF export.")
        else:
            st.caption("No messages yet to export.")

# ---------------------------------------------------------------------------
# Phase 5 — top-level navigation. Sidebar above is completely unchanged;
# these tabs organize the main content area only.
# ---------------------------------------------------------------------------
tab_chat, tab_docs, tab_insights, tab_settings = st.tabs(
    ["💬 Chat", "📚 Document Intelligence", "📈 Insights Dashboard", "⚙ Settings"]
)

# ===========================================================================
# CHAT — kept deliberately minimal per spec: confidence/model/timing badges,
# like/dislike/copy/regenerate, and suggested follow-ups. No sources
# expander, no analytics, no evaluation — those moved to the other tabs.
# ===========================================================================
with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            "<div class='glass-card'>"
            "<b>👋 Welcome!</b> Upload one or more insurance policy PDFs in the sidebar, "
            "click <b>Build KB</b>, then ask a question below.<br><br>"
            "<b>Supported policies:</b> CMCHIS · PM-JAY · ESIC · CGHS · Private insurers<br>"
            "<b>Tips:</b> Ask follow-up questions naturally — the assistant remembers context. "
            "Press <b>Enter</b> to send.<br>"
            "<b>Quick actions:</b> use the sidebar's Conversations panel to rename, search, or start a new thread."
            "</div>",
            unsafe_allow_html=True,
        )
        example_picked = render_suggested_questions(
            ["What does CMCHIS cover for surgeries?", "Am I eligible for PM-JAY?", "What is the claim process?"],
            key_prefix="example",
        )
    else:
        example_picked = None

    for i, msg in enumerate(st.session_state.messages):
        render_chat_message(msg["role"], msg["content"])
        render_timestamp(msg.get("timestamp", ""))
        if msg["role"] == "assistant" and msg.get("sources"):
            render_answer_meta(msg["confidence"], msg["provider"], msg["model"], msg["retrieval_time"], msg["generation_time"])
            prior_user = next((m["content"] for m in reversed(st.session_state.messages[:i]) if m["role"] == "user"), "")
            action = render_message_actions(i, msg["content"])
            if st.session_state.feedback.get(i):
                st.caption(f"You marked this: {st.session_state.feedback[i]}")
            if action in ("like", "dislike"):
                st.session_state.feedback[i] = action
                st.toast(f"Feedback recorded: {action}", icon="👍" if action == "like" else "👎")
            elif action == "regenerate":
                st.session_state.messages = st.session_state.messages[:i - 1] if i > 0 else []
                st.session_state.pending_suggestion = prior_user
                st.rerun()
            suggestions = generate_suggestions(msg.get("intent", "General"), prior_user)
            picked_followup = render_suggested_questions(suggestions, key_prefix=f"followup_{i}") if suggestions else None
            if picked_followup:
                st.session_state.pending_suggestion = picked_followup

    if st.session_state.stream_state and st.session_state.stream_state.get("active"):
        with st.chat_message("assistant"):
            render_streaming_fragment()

    query = st.chat_input("Ask a question about the uploaded insurance policies...")
    final_query = query or st.session_state.pending_suggestion or example_picked
    st.session_state.pending_suggestion = None

    if final_query:
        if not st.session_state.vectorstore:
            st.error("Please build the knowledge base first by uploading PDFs and clicking 'Build KB'.")
        elif st.session_state.stream_state and st.session_state.stream_state.get("active"):
            st.warning("Please wait for the current response to finish (or click Stop).")
        else:
            final_query = sanitize_query(final_query)
            ts = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({"role": "user", "content": final_query, "timestamp": ts})
            conversation_store.save_messages(st.session_state.conv_id, st.session_state.messages)
            try:
                _start_streaming(final_query, provider, top_k)
            except Exception as exc:
                st.error(safe_error_message(exc))
            st.rerun()

# ===========================================================================
# DOCUMENT INTELLIGENCE — all business logic lives in utils/document_explorer.py,
# utils/policy_parser.py, utils/document_compare.py, utils/document_search.py.
# ===========================================================================
with tab_docs:
    from utils.vectorstore import INDEX_DIR as _DI_INDEX_DIR
    from utils.document_explorer import (
        render_document_explorer, render_document_statistics,
        render_duplicate_detection, render_kb_health,
    )
    from utils.document_search import search_with_highlighting
    from utils.policy_parser import generate_policy_summary, extract_clauses
    from utils.document_compare import compare_policies

    UPLOAD_DIR = "data/uploads"
    doc_sub = st.tabs([
        "🗂 Explorer", "🔍 Search", "📊 Statistics", "📝 Policy Summary", "⚖ Compare",
        "🔑 Clause Extraction", "♻ Duplicates", "🩺 Coverage Cards", "🔬 Source Inspector", "❤ KB Health",
    ])

    with doc_sub[0]:
        render_document_explorer(st.session_state.all_chunks, UPLOAD_DIR)

    with doc_sub[1]:
        term = st.text_input("Search across all uploaded documents")
        if term:
            for hit in search_with_highlighting(st.session_state.all_chunks, term):
                st.markdown(f"**{hit['source']} — chunk {hit['chunk_index']}**", unsafe_allow_html=True)
                st.markdown(hit.get("highlighted_snippet", hit["snippet"]), unsafe_allow_html=True)

    with doc_sub[2]:
        render_document_statistics(st.session_state.all_chunks)

    with doc_sub[3]:
        st.caption("One LLM call per document — generated on demand, not automatically.")
        sources = sorted({c.metadata.get("source") for c in st.session_state.all_chunks})
        chosen = st.selectbox("Select a document to summarize", options=sources or ["No documents"])
        if sources and st.button("Generate Summary"):
            doc_text = "\n".join(c.page_content for c in st.session_state.all_chunks if c.metadata.get("source") == chosen)
            with st.spinner("Generating summary..."):
                summary = generate_policy_summary(doc_text, provider=provider)
            for label, value in summary["sections"].items():
                st.markdown(f"**{label}:** {value}")

    with doc_sub[4]:
        st.caption("One LLM call comparing two documents — generated on demand.")
        sources = sorted({c.metadata.get("source") for c in st.session_state.all_chunks})
        if len(sources) >= 2:
            col1, col2 = st.columns(2)
            doc_a = col1.selectbox("Document A", options=sources, key="compare_a")
            doc_b = col2.selectbox("Document B", options=sources, index=min(1, len(sources) - 1), key="compare_b")
            if st.button("Compare Policies"):
                text_a = "\n".join(c.page_content for c in st.session_state.all_chunks if c.metadata.get("source") == doc_a)
                text_b = "\n".join(c.page_content for c in st.session_state.all_chunks if c.metadata.get("source") == doc_b)
                with st.spinner("Comparing..."):
                    comparison = compare_policies(doc_a, text_a, doc_b, text_b, provider=provider)
                st.markdown(comparison["comparison_markdown"])
        else:
            st.info("Upload at least two documents to compare.")

    with doc_sub[5]:
        st.caption("One LLM call per document — extracts standard clause categories.")
        sources = sorted({c.metadata.get("source") for c in st.session_state.all_chunks})
        chosen_clause_doc = st.selectbox("Select a document", options=sources or ["No documents"], key="clause_doc")
        if sources and st.button("Extract Clauses"):
            doc_text = "\n".join(c.page_content for c in st.session_state.all_chunks if c.metadata.get("source") == chosen_clause_doc)
            with st.spinner("Extracting clauses..."):
                clauses = extract_clauses(doc_text, provider=provider)
            for label, value in clauses["sections"].items():
                st.markdown(f"**{label}:** {value}")

    with doc_sub[6]:
        render_duplicate_detection(st.session_state.all_chunks, UPLOAD_DIR)

    with doc_sub[7]:
        st.caption("Reuses the last generated Policy Summary for each document, if available.")
        if "policy_summaries_cache" not in st.session_state:
            st.session_state.policy_summaries_cache = {}
        sources = sorted({c.metadata.get("source") for c in st.session_state.all_chunks})
        for src in sources:
            cached = st.session_state.policy_summaries_cache.get(src)
            if not cached:
                st.markdown(f"<div class='glass-card'>No summary generated yet for <b>{src}</b>. "
                             f"Visit the Policy Summary tab first.</div>", unsafe_allow_html=True)
                continue
            sections = cached["sections"]
            st.markdown(
                f"<div class='glass-card'><b>{src}</b><br>"
                f"<b>Coverage:</b> {sections.get('Coverage', 'n/a')}<br>"
                f"<b>Benefits:</b> {sections.get('Benefits', 'n/a')}<br>"
                f"<b>Waiting Period:</b> {sections.get('Waiting Period', 'n/a')}<br>"
                f"<b>Exclusions:</b> {sections.get('Exclusions', 'n/a')}</div>",
                unsafe_allow_html=True,
            )

    with doc_sub[8]:
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant" and m.get("retrieved_chunks")]
        if not assistant_msgs:
            st.info("Ask a question in Chat first, then inspect its sources here.")
        else:
            msg_labels = [f"#{i+1}: {m['content'][:50]}..." for i, m in enumerate(assistant_msgs)]
            picked_idx = st.selectbox("Select an answer to inspect", options=range(len(assistant_msgs)), format_func=lambda i: msg_labels[i])
            for chunk in assistant_msgs[picked_idx]["retrieved_chunks"]:
                st.markdown(
                    f"<div class='glass-card'><b>{chunk['source']}</b> — chunk {chunk['chunk_index']}<br>"
                    f"Similarity (RRF): {chunk.get('rrf_score', 'n/a')} · Reranker score: {chunk.get('rerank_score', 'n/a')} "
                    f"· Final rank: {chunk.get('final_rank', 'n/a')}<br><br>{chunk['content'][:600]}</div>",
                    unsafe_allow_html=True,
                )

    with doc_sub[9]:
        render_kb_health(st.session_state.vectorstore, st.session_state.bm25_index, st.session_state.all_chunks, _DI_INDEX_DIR)

# ===========================================================================
# INSIGHTS DASHBOARD — all four sections reuse existing Phase 3/4 data
# (query_log, evaluations) collected silently; this is a new display only.
# ===========================================================================
with tab_insights:
    from utils.insights import (
        render_analytics_section, render_evaluation_section,
        render_performance_section, render_knowledge_base_section,
    )
    insight_sub = st.tabs(["Analytics", "Evaluation", "Performance", "Knowledge Base"])
    with insight_sub[0]:
        render_analytics_section(st.session_state.query_log)
    with insight_sub[1]:
        render_evaluation_section(st.session_state.evaluations)
    with insight_sub[2]:
        render_performance_section(st.session_state.query_log, memory)
    with insight_sub[3]:
        from utils.vectorstore import INDEX_DIR as _INS_INDEX_DIR
        render_knowledge_base_section(st.session_state.vectorstore, st.session_state.bm25_index,
                                       st.session_state.all_chunks, _INS_INDEX_DIR)

# ===========================================================================
# SETTINGS — read-only mirror of live sidebar controls (which remain the
# source of truth, per "do not remove existing widgets") plus additional
# controls that don't exist in the sidebar: developer options, log level.
# ===========================================================================
with tab_settings:
    st.markdown("#### Current Configuration")
    st.caption("LLM Provider, Theme, and Retrieval Settings are controlled live in the sidebar — shown here read-only for reference.")
    st.markdown(
        f"**LLM Provider:** {provider}  \n"
        f"**Embedding Model:** {EMBEDDING_MODEL_NAME}  \n"
        f"**Theme:** {st.session_state.theme.title()}  \n"
        f"**Top-K:** {top_k}  \n"
        f"**LangSmith Tracing:** {'Enabled' if is_tracing_enabled() else 'Disabled'}  \n"
        f"**Reranker Mode:** {reranker_mode()}"
    )

    st.markdown("---")
    st.markdown("#### Export Options")
    st.caption("Available formats: Markdown, JSON (with metadata), PDF — see the sidebar's Export Conversation panel.")

    st.markdown("---")
    st.markdown("#### Security")
    from config import CONFIG as _SETTINGS_CONFIG
    st.markdown(
        f"**Max upload size:** {_SETTINGS_CONFIG.security.max_file_size_mb}MB  \n"
        f"**Allowed file types:** {', '.join(_SETTINGS_CONFIG.security.allowed_extensions)}  \n"
        f"**Max files per upload:** {_SETTINGS_CONFIG.security.max_files_per_upload}"
    )

    st.markdown("---")
    st.markdown("#### Conversation Memory")
    st.markdown(
        f"**Max turns kept verbatim:** {_SETTINGS_CONFIG.memory.max_turns}  \n"
        f"**Summarize after:** {_SETTINGS_CONFIG.memory.summarize_after_turns} turns  \n"
        f"**Current turns:** {len(memory.turns)}"
    )

    st.markdown("---")
    st.markdown("#### Developer Options")
    st.session_state.setdefault("dev_show_raw_state", False)
    st.session_state.dev_show_raw_state = st.checkbox(
        "Show raw session state (debugging)", value=st.session_state.dev_show_raw_state,
    )
    if st.session_state.dev_show_raw_state:
        st.json({
            "conv_id": st.session_state.conv_id,
            "num_messages": len(st.session_state.messages),
            "num_chunks": len(st.session_state.all_chunks),
            "bm25_active": st.session_state.bm25_index is not None,
        })
    log_path = os.path.join(os.path.dirname(__file__), "logs", "app.log")
    st.caption(f"Logs are written to: {log_path}")
# -------------------------------------------------
# Footer
# -------------------------------------------------
render_footer()
