"""
utils/rag_chain.py
Phase 3 — orchestrates the full enterprise pipeline:
  Query -> Intent Classification -> Hybrid Retrieval (FAISS + BM25 + RRF)
        -> Re-ranking -> Structured Prompt (+ conversational memory)
        -> LLM (via model_router) -> Hallucination Check -> Structured Answer
Phase 1/2 retrieval primitives (vectorstore.py similarity search) are reused
unchanged inside hybrid_retrieval.py; nothing here removes FAISS-only search.
"""

import time
from typing import List, Dict, Any, Optional
from utils.hybrid_retrieval import hybrid_search
from utils.reranker import rerank, reranker_mode
from utils.model_router import route_llm
from utils.query_classifier import classify_intent
from utils.hallucination_check import check_groundedness
from utils.memory import ConversationMemory
from utils.tracing import traceable, is_tracing_enabled
from utils.logger import get_logger
from config import CONFIG

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an Insurance Policy Assistant for Indian health insurance \
schemes (CMCHIS, PM-JAY, ESIC, CGHS, and private insurers).

STRICT RULES:
1. Answer ONLY using the provided context chunks below. Never use outside knowledge.
2. If the answer is not present in the context, respond exactly with:
   "Information not found in documents."
3. Do not speculate, infer, or fabricate insurance rules, amounts, or eligibility.
4. Use the conversation history only to resolve pronouns/follow-ups — never as a
   source of factual claims.
5. Structure your answer using these labeled sections, omitting any section that
   has no relevant information in the context:

Answer: <direct answer>
Policy Name: <if identifiable>
Applicable Sections: <section/clause references if present>
Eligibility: <if relevant>
Benefits: <if relevant>
Premium Information: <if relevant>
Waiting Period: <if relevant>
Claim Process: <if relevant>
Important Notes: <caveats>

CONVERSATION HISTORY:
{history}

CONTEXT:
{context}

USER QUESTION:
{question}
"""


def _format_context(scored_docs) -> str:
    parts = []
    for item in scored_docs:
        doc = item["document"]
        src = doc.metadata.get("source", "unknown")
        idx = doc.metadata.get("chunk_index", "?")
        parts.append(f"[{src} | chunk {idx}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _parse_structured_sections(answer_text: str) -> Dict[str, str]:
    labels = [
        "Answer", "Policy Name", "Applicable Sections", "Eligibility", "Benefits",
        "Premium Information", "Waiting Period", "Claim Process", "Important Notes",
    ]
    sections = {}
    remaining = answer_text
    for i, label in enumerate(labels):
        marker = f"{label}:"
        if marker not in remaining:
            continue
        start = remaining.index(marker) + len(marker)
        end = len(remaining)
        for next_label in labels[i + 1:]:
            next_marker = f"{next_label}:"
            if next_marker in remaining[start:]:
                end = start + remaining[start:].index(next_marker)
                break
        sections[label] = remaining[start:end].strip()
    return sections


def _confidence_from_scores(scored_docs) -> str:
    if not scored_docs:
        return "Low"
    avg = sum(item["rerank_score"] for item in scored_docs) / len(scored_docs)
    if avg > 0.55:
        return "High"
    elif avg > 0.25:
        return "Medium"
    return "Low"


@traceable(name="hybrid_retrieval_and_rerank", run_type="retriever")
def _retrieve_and_rerank(vectorstore, bm25_index, bm25_corpus, question: str):
    fused = hybrid_search(vectorstore, bm25_index, bm25_corpus, question,
                           k=CONFIG.retrieval.top_k_initial)
    return rerank(question, fused, top_n=CONFIG.retrieval.top_k_final)


def prepare_retrieval(
    vectorstore, question: str, bm25_index=None, bm25_corpus: Optional[list] = None,
    memory: Optional[ConversationMemory] = None,
) -> Dict[str, Any]:
    """
    Phase 4 addition — retrieval + prompt construction only, without calling
    the LLM. Lets app.py stream the generation step live while reusing the
    exact same retrieval/reranking/prompt logic as answer_question(). Returns
    everything finalize_answer() needs to complete the pipeline afterward.
    """
    intent = classify_intent(question)
    retrieval_start = time.time()
    scored_docs = _retrieve_and_rerank(vectorstore, bm25_index, bm25_corpus, question)
    retrieval_time = round(time.time() - retrieval_start, 2)

    prompt = None
    context = ""
    if scored_docs:
        context = _format_context(scored_docs)
        history = memory.context_string() if memory else ""
        prompt = SYSTEM_PROMPT.format(history=history or "None", context=context, question=question)

    return {
        "intent": intent, "scored_docs": scored_docs, "context": context,
        "prompt": prompt, "retrieval_time": retrieval_time,
    }


def finalize_answer(
    question: str, prepared: Dict[str, Any], answer_text: str,
    provider: str, model: str, generation_time: float,
) -> Dict[str, Any]:
    """
    Phase 4 addition — post-processing only (groundedness, structured-section
    parsing, confidence, source formatting). Takes the already-generated
    (possibly streamed) answer text and produces the exact same result shape
    answer_question() has always returned, so existing callers/UI code don't
    need to know whether the text came from a streamed or blocking call.
    """
    scored_docs = prepared["scored_docs"]
    intent = prepared["intent"]
    retrieval_time = prepared["retrieval_time"]

    if not scored_docs:
        return {
            "answer": "Information not found in documents.",
            "sections": {}, "sources": [], "confidence": "Low", "intent": intent,
            "retrieved_chunks": [], "provider": provider, "model": "n/a",
            "retrieval_time": retrieval_time, "generation_time": 0.0,
            "grounded": True, "reranker_mode": reranker_mode(), "traced": is_tracing_enabled(),
        }

    context = prepared["context"]
    hallucination_report = check_groundedness(answer_text, context)
    if not hallucination_report["grounded"]:
        logger.warning("Groundedness check flagged %d unsupported sentence(s)",
                        len(hallucination_report["flagged_sentences"]))

    sections = _parse_structured_sections(answer_text)

    sources = [
        {"document": item["document"].metadata.get("source", "unknown"),
         "chunk_index": item["document"].metadata.get("chunk_index", "?")}
        for item in scored_docs
    ]

    return {
        "answer": sections.get("Answer", answer_text),
        "sections": sections,
        "sources": sources,
        "confidence": _confidence_from_scores(scored_docs),
        "intent": intent,
        "retrieved_chunks": [
            {"content": item["document"].page_content,
             "source": item["document"].metadata.get("source"),
             "chunk_index": item["document"].metadata.get("chunk_index"),
             "score": item["rerank_score"],  # backward-compat key for unmodified Phase 2 UI
             "rrf_score": item["rrf_score"], "rerank_score": item["rerank_score"],
             "final_rank": item["final_rank"]}
            for item in scored_docs
        ],
        "provider": provider,
        "model": model,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "grounded": hallucination_report["grounded"],
        "groundedness_score": hallucination_report["overlap_ratio"],
        "reranker_mode": reranker_mode(),
        "traced": is_tracing_enabled(),
    }


@traceable(name="insurance_rag_answer_v3", run_type="chain")
def answer_question(
    vectorstore, question: str, top_k: int = 4, provider: str = "OpenAI",
    bm25_index=None, bm25_corpus: Optional[list] = None,
    memory: Optional[ConversationMemory] = None,
) -> Dict[str, Any]:
    """
    Full Phase 3/4 pipeline, unchanged signature and return shape. `top_k` is
    retained for UI compatibility and maps to CONFIG.retrieval.top_k_final if
    not overridden by the sidebar slider. Internally now composed of
    prepare_retrieval() + a blocking LLM call + finalize_answer(), so the
    non-streaming path (used by "Regenerate" and any future batch use) and
    the Phase 4 streaming path in app.py share 100% of the same logic with
    no duplication.
    """
    prepared = prepare_retrieval(vectorstore, question, bm25_index, bm25_corpus, memory)
    if not prepared["scored_docs"]:
        return finalize_answer(question, prepared, "", provider, "n/a", 0.0)

    llm_result = route_llm(provider=provider, prompt=prepared["prompt"], context=prepared["context"])
    return finalize_answer(
        question, prepared, llm_result["content"],
        llm_result["provider"], llm_result["model"], llm_result["latency_seconds"],
    )
