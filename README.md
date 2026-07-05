# Insurance Policy RAG Assistant — Phase 3

Enterprise backend upgrade of the Phase 2 project. **Frontend is unchanged**:
`app.py` layout, `utils/ui_components.py`, and `utils/style.css` keep their
exact widgets, sidebar order, CSS, and behavior. `utils/model_router.py` and
`utils/tracing.py` are untouched (your custom model IDs and LangSmith project
name are preserved as-is).

## Project Structure
```
insurance_rag_assistant/
├── app.py                      # UNCHANGED layout; Clear KB now also resets
│                                  memory; uploads validated; queries sanitized
├── config.py                    # NEW: central dataclass configuration
├── requirements.txt              # + rank-bm25
├── data/{uploads, vectorstore}    # your existing PDF + FAISS index, preserved
└── utils/
    ├── loader.py, chunker.py, embeddings.py, vectorstore.py   # UNCHANGED
    ├── model_router.py, tracing.py                              # UNCHANGED
    ├── style.css, ui_components.py                                # UNCHANGED
    ├── rag_chain.py                # REWRITTEN: same answer_question(vectorstore,
    │                                  question, top_k, provider) signature;
    │                                  new bm25_index/bm25_corpus/memory kwargs
    │                                  default to None so old call sites still
    │                                  work; retrieved_chunks keeps its original
    │                                  'score' key for the unmodified UI
    ├── hybrid_retrieval.py           # NEW: FAISS + BM25 via Reciprocal Rank Fusion
    ├── reranker.py                    # NEW: real cross-encoder (sentence-transformers,
    │                                     already in your requirements.txt) with
    │                                     automatic lexical-overlap fallback if absent
    ├── query_classifier.py             # NEW: intent detection (Eligibility, Claims, etc.)
    ├── hallucination_check.py           # NEW: groundedness heuristic
    ├── memory.py                         # NEW: conversational memory + summarization
    ├── analytics.py                       # NEW: KPI aggregation (pandas)
    ├── evaluation.py                       # NEW: heuristic RAG quality metrics
    ├── document_intelligence.py             # NEW: per-document stats, search-within
    ├── export_utils.py                       # NEW: conversation export (Markdown/PDF)
    ├── security.py                            # NEW: upload validation, input sanitization
    ├── stats.py                                # NEW: session/response statistics
    └── logger.py                                # NEW: centralized rotating file logging
```

## Run Instructions
```bash
pip install -r requirements.txt
streamlit run app.py
```
Your existing `data/vectorstore/` index and uploaded PDF are preserved and
will load automatically on first run — no need to rebuild the knowledge base.

## What Changed vs. What Didn't
- **UI is byte-for-byte unchanged**: same sidebar order, same buttons, same
  CSS, same chat bubbles. The existing "Clear KB" button now also resets
  conversation memory — no new widget was added.
- **`answer_question()` keeps its original signature and return keys**
  (`answer`, `sources`, `confidence`, `retrieved_chunks`, `provider`, `model`,
  `retrieval_time`, `generation_time`) so nothing calling it needs to change.
  `retrieved_chunks` items keep the original `score` key (now the rerank
  score) so `render_sources_expander` works with zero modifications.
- **Your `model_router.py` model IDs and `tracing.py` project name are
  exactly as you had them** — neither file was touched.
- Uploads now pass through file type/size/count validation; queries are
  sanitized before use; both fail gracefully with `st.error`/`st.warning`,
  the same UI pattern Phase 2 already used.
- `analytics.py` and `evaluation.py` populate `st.session_state.query_log` /
  `st.session_state.evaluations` every turn, ready for a future dashboard —
  not displayed yet, since that would require new UI.

## Honest Scope Notes
- Cross-encoder re-ranking uses the real `sentence-transformers` model since
  it's already in your `requirements.txt`; if it's ever uninstalled,
  `reranker.py` automatically falls back to a lexical-overlap heuristic
  instead of crashing.
- Hallucination detection and evaluation metrics are word-overlap heuristics,
  not a trained NLI model or RAGAS — labeled as such in `evaluation.py`'s
  docstring.
- Query intent classification is keyword-based, not a trained classifier.

## Verified Before Delivery
- Full syntax compile check on every file.
- Full import check against your actual `model_router.py` and `tracing.py`
  (not a generic stand-in) — confirmed your custom model IDs and debug output
  still print correctly.
- Hybrid retrieval → RRF → rerank pipeline smoke-tested against a synthetic
  knowledge base using your real project files.
