import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import AbstractSet, Any, Dict, List, Sequence

from app.core.config import settings
from app.retrieval.bm25 import BM25Index, BM25Retriever
from app.services.embeddings import STOP_WORDS, TOKEN_PATTERN, tokenize
from app.services.llm_client import (
    ChatCompletionResult,
    LLMClientError,
    chat_completion,
    is_llm_configured,
)
from app.security.defenses import BASELINE_SECURITY_MODE
from app.services.prompts import build_rag_messages
from app.services.reranker import rerank_chunks
from app.services.storage_paths import get_document_storage_paths
from app.services.text_loader import ExtractedSection
from app.services.text_splitter import split_text
from app.services.vector_store import index_vector_chunks, search_vector_chunks


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    return tokenize(text)


def _default_chunks_file() -> Path:
    return get_document_storage_paths().chunks_file


def _default_vectors_file() -> Path:
    return get_document_storage_paths().vectors_file


def _resolve_chunks_file(index_path: Path | None) -> Path:
    return index_path or _default_chunks_file()


def _resolve_vectors_file(vector_index_path: Path | None) -> Path:
    return vector_index_path or _default_vectors_file()


def _load_chunks(index_path: Path | None = None) -> List[Dict[str, Any]]:
    index_path = _resolve_chunks_file(index_path)
    if not index_path.exists():
        return []

    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_chunks(chunks: List[Dict[str, Any]], index_path: Path | None = None) -> None:
    index_path = _resolve_chunks_file(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _authorized_chunks(
    chunks: Sequence[Dict[str, Any]],
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in chunks
        if isinstance(chunk.get("document_id"), str)
        and bool(chunk["document_id"])
        and chunk["document_id"] in allowed_document_ids
    ]


def build_bm25_retriever(
    index_path: Path | None = None,
    *,
    allowed_document_ids: AbstractSet[str],
) -> BM25Retriever:
    """Build an independent lexical retriever from the currently stored chunks."""
    return BM25Retriever(
        BM25Index(_authorized_chunks(_load_chunks(index_path), allowed_document_ids))
    )


def search_bm25_chunks(
    query: str,
    top_k: int = 5,
    index_path: Path | None = None,
    *,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    """Retrieve stored chunks by BM25 without changing existing search modes."""
    return build_bm25_retriever(
        index_path,
        allowed_document_ids=allowed_document_ids,
    ).retrieve(query, top_k)


def index_document(
    document_id: str,
    filename: str,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    index_path: Path | None = None,
    vector_index_path: Path | None = None,
    sections: Sequence[ExtractedSection] | None = None,
) -> List[Dict[str, Any]]:
    """
    Split a document into chunks and persist them in the local knowledge base.
    Existing chunks for the same document are replaced to keep upload retries clean.
    """
    index_path = _resolve_chunks_file(index_path)
    vector_index_path = _resolve_vectors_file(vector_index_path)
    chunks = _load_chunks(index_path)
    chunks = [chunk for chunk in chunks if chunk["document_id"] != document_id]

    source_sections = sections or (ExtractedSection(text=text),)
    new_chunks = []
    position = 1
    for section in source_sections:
        for content in split_text(section.text, chunk_size, chunk_overlap):
            tokens = _tokenize(content)
            new_chunks.append(
                {
                    "chunk_id": f"{document_id}-{position}",
                    "document_id": document_id,
                    "filename": filename,
                    "position": position,
                    "chunk_index": position - 1,
                    "page_number": section.page_number,
                    "content": content,
                    "token_count": len(tokens),
                    "created_at": _now_iso(),
                }
            )
            position += 1

    _save_chunks(chunks + new_chunks, index_path)
    index_vector_chunks(document_id, new_chunks, vector_index_path)
    return new_chunks


def list_documents(index_path: Path | None = None) -> List[Dict[str, Any]]:
    documents: Dict[str, Dict[str, Any]] = {}

    for chunk in _load_chunks(index_path):
        document_id = chunk["document_id"]
        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": chunk["filename"],
                "chunk_count": 0,
                "created_at": chunk.get("created_at"),
            }
        documents[document_id]["chunk_count"] += 1

    return sorted(documents.values(), key=lambda item: item.get("created_at") or "", reverse=True)


def get_all_chunks(index_path: Path | None = None) -> List[Dict[str, Any]]:
    """Return stored chunks for explicit maintenance tasks such as W10 backfill."""
    return _load_chunks(index_path)


def get_document_chunks(document_id: str, index_path: Path | None = None) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in sorted(_load_chunks(index_path), key=lambda item: item["position"])
        if chunk["document_id"] == document_id
    ]


def _cosine_similarity(left: Counter, right: Counter) -> float:
    common_tokens = set(left) & set(right)
    dot_product = sum(left[token] * right[token] for token in common_tokens)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


GENERIC_QUESTION_TERMS = {
    "allowed",
    "answer",
    "automatically",
    "carrying",
    "company",
    "current",
    "document",
    "documents",
    "exactly",
    "immediate",
    "indexed",
    "knowledge",
    "member",
    "policy",
    "question",
    "questions",
    "relevant",
    "reliably",
    "right",
    "rules",
    "still",
    "while",
    "without",
}
EXACT_EVIDENCE_TERMS = {
    "attempts",
    "bereavement",
    "canada",
    "contractor",
    "contractors",
    "crm",
    "dental",
    "gym",
    "lock",
    "membership",
    "maternity",
    "parking",
    "purchase",
    "slack",
    "sso",
    "stock",
}


def _raw_salient_tokens(text: str) -> List[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(text):
        normalized = token.lower()
        if normalized in STOP_WORDS or normalized in GENERIC_QUESTION_TERMS:
            continue
        if len(normalized) < 6 and normalized not in EXACT_EVIDENCE_TERMS:
            continue
        tokens.append(normalized)
    return tokens


def _has_unsupported_salient_terms(question: str, chunks: List[Dict[str, Any]]) -> bool:
    if not chunks:
        return False

    top_score = float(chunks[0].get("score") or 0.0)
    if top_score >= 0.35:
        return False

    context_text = " ".join(chunk.get("content", "") for chunk in chunks).lower()
    missing_terms = [
        token
        for token in _raw_salient_tokens(question)
        if token not in context_text
    ]
    return any(token in EXACT_EVIDENCE_TERMS for token in missing_terms)


def _search_keyword_chunks(
    question: str,
    top_k: int,
    index_path: Path,
    min_score: float,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    query_vector = Counter(_tokenize(question))
    scored_chunks = []

    for chunk in _authorized_chunks(_load_chunks(index_path), allowed_document_ids):
        chunk_vector = Counter(_tokenize(chunk["content"]))
        score = _cosine_similarity(query_vector, chunk_vector)
        if score > min_score:
            scored_chunks.append({**chunk, "score": round(score, 4), "retrieval_mode": "keyword"})

    scored_chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
    return scored_chunks[:top_k]


def _search_hybrid_chunks(
    question: str,
    top_k: int,
    index_path: Path,
    vector_index_path: Path,
    min_score: float,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    keyword_candidates = _search_keyword_chunks(
        question,
        top_k=max(top_k * 4, 20),
        index_path=index_path,
        min_score=0.0,
        allowed_document_ids=allowed_document_ids,
    )
    vector_candidates = search_vector_chunks(
        question,
        top_k=max(top_k * 4, 20),
        index_path=vector_index_path,
        min_score=0.0,
        allowed_document_ids=allowed_document_ids,
    )

    merged: Dict[str, Dict[str, Any]] = {}

    for chunk in keyword_candidates:
        chunk_id = chunk["chunk_id"]
        merged[chunk_id] = {
            **chunk,
            "keyword_score": chunk["score"],
            "vector_score": 0.0,
        }

    for chunk in vector_candidates:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in merged:
            merged[chunk_id] = {
                **chunk,
                "keyword_score": 0.0,
                "vector_score": chunk["score"],
            }
        else:
            merged[chunk_id]["vector_score"] = chunk["score"]
            if "embedding_model" in chunk:
                merged[chunk_id]["embedding_model"] = chunk.get("embedding_model")
            if "query_embedding_model" in chunk:
                merged[chunk_id]["query_embedding_model"] = chunk.get("query_embedding_model")

    scored_chunks = []
    for chunk in merged.values():
        hybrid_score = (
            settings.hybrid_keyword_weight * chunk["keyword_score"]
            + settings.hybrid_vector_weight * chunk["vector_score"]
        )
        if hybrid_score > 0:
            scored_chunks.append(
                {
                    **chunk,
                    "score": round(hybrid_score, 4),
                    "hybrid_score": round(hybrid_score, 4),
                    "keyword_score": round(chunk["keyword_score"], 4),
                    "vector_score": round(chunk["vector_score"], 4),
                    "retrieval_mode": "hybrid",
                }
            )

    scored_chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
    return _select_hybrid_context_candidates(scored_chunks, top_k, min_score)


def _select_hybrid_context_candidates(
    scored_chunks: List[Dict[str, Any]],
    top_k: int,
    min_score: float,
) -> List[Dict[str, Any]]:
    selected = [chunk for chunk in scored_chunks if chunk["score"] > min_score][:top_k]
    if len(selected) >= top_k or not scored_chunks:
        return selected

    selected_chunk_ids = {chunk["chunk_id"] for chunk in selected}
    selected_document_ids = {chunk["document_id"] for chunk in selected}
    relaxed_min_score = min_score * 0.75

    for chunk in scored_chunks:
        if len(selected) >= top_k:
            break
        if chunk["chunk_id"] in selected_chunk_ids:
            continue
        if chunk["document_id"] in selected_document_ids:
            continue
        if chunk["score"] <= relaxed_min_score:
            continue

        selected.append(chunk)
        selected_chunk_ids.add(chunk["chunk_id"])
        selected_document_ids.add(chunk["document_id"])

    return selected


def _expand_adjacent_chunks(
    chunks: List[Dict[str, Any]],
    index_path: Path | None,
    window: int,
    max_contexts: int,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    if not chunks or window <= 0 or max_contexts <= 0:
        return chunks

    all_chunks = _authorized_chunks(_load_chunks(index_path), allowed_document_ids)
    chunks_by_location = {
        (chunk["document_id"], chunk["position"]): chunk
        for chunk in all_chunks
    }
    expanded: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for chunk in chunks:
        if len(expanded) >= max_contexts:
            break

        chunk_id = chunk["chunk_id"]
        if chunk_id not in seen_chunk_ids:
            expanded.append({**chunk, "context_role": chunk.get("context_role", "retrieved")})
            seen_chunk_ids.add(chunk_id)

        document_id = chunk["document_id"]
        position = chunk["position"]
        for offset in range(1, window + 1):
            for adjacent_position in (position - offset, position + offset):
                if len(expanded) >= max_contexts:
                    break

                adjacent = chunks_by_location.get((document_id, adjacent_position))
                if not adjacent or adjacent["chunk_id"] in seen_chunk_ids:
                    continue

                expanded.append(
                    {
                        **adjacent,
                        "score": chunk.get("score"),
                        "expanded_from_chunk_id": chunk_id,
                        "context_role": "adjacent",
                        "retrieval_mode": chunk.get("retrieval_mode", "context_expansion"),
                    }
                )
                seen_chunk_ids.add(adjacent["chunk_id"])

    return expanded


def finalize_contexts(
    question: str,
    chunks: List[Dict[str, Any]],
    index_path: Path | None,
    *,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    """Apply the production context-expansion and evidence-filtering policy."""
    authorized_chunks = _authorized_chunks(chunks, allowed_document_ids)
    expanded = _expand_adjacent_chunks(
        authorized_chunks,
        index_path,
        settings.context_window_chunks,
        settings.max_expanded_contexts,
        allowed_document_ids,
    )
    if _has_unsupported_salient_terms(question, expanded):
        return []
    return expanded


def build_citations(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    seen_chunk_ids = set()

    for context in contexts:
        chunk_id = context.get("chunk_id")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue

        citations.append(
            {
                "chunk_id": chunk_id,
                "document_id": context.get("document_id"),
                "filename": context.get("filename"),
                "position": context.get("position"),
                "chunk_index": context.get("chunk_index"),
                "page_number": context.get("page_number"),
                "score": context.get("score"),
                "retrieval_mode": context.get("retrieval_mode"),
                "context_role": context.get("context_role", "retrieved"),
                "expanded_from_chunk_id": context.get("expanded_from_chunk_id"),
            }
        )
        seen_chunk_ids.add(chunk_id)

    return citations


def search_chunks(
    question: str,
    top_k: int = 3,
    index_path: Path | None = None,
    retrieval_mode: str = "keyword",
    vector_index_path: Path | None = None,
    min_score: float | None = None,
    *,
    allowed_document_ids: AbstractSet[str],
) -> List[Dict[str, Any]]:
    index_path = _resolve_chunks_file(index_path)
    vector_index_path = _resolve_vectors_file(vector_index_path)
    score_threshold = min_score
    if score_threshold is None:
        if retrieval_mode == "vector":
            score_threshold = settings.vector_min_score
        elif retrieval_mode == "hybrid":
            score_threshold = settings.hybrid_min_score
        elif retrieval_mode == "rerank":
            score_threshold = settings.rerank_min_score
        else:
            score_threshold = settings.keyword_min_score

    if retrieval_mode == "vector":
        chunks = search_vector_chunks(
            question,
            top_k,
            vector_index_path,
            min_score=score_threshold,
            allowed_document_ids=allowed_document_ids,
        )
        return finalize_contexts(
            question,
            chunks,
            index_path,
            allowed_document_ids=allowed_document_ids,
        )

    if retrieval_mode == "hybrid":
        chunks = _search_hybrid_chunks(
            question,
            top_k,
            index_path,
            vector_index_path,
            score_threshold,
            allowed_document_ids,
        )
        return finalize_contexts(
            question,
            chunks,
            index_path,
            allowed_document_ids=allowed_document_ids,
        )

    if retrieval_mode == "rerank":
        candidates = _search_hybrid_chunks(
            question,
            top_k=max(top_k * settings.rerank_candidate_multiplier, top_k),
            index_path=index_path,
            vector_index_path=vector_index_path,
            min_score=settings.hybrid_min_score,
            allowed_document_ids=allowed_document_ids,
        )
        chunks = rerank_chunks(question, candidates, top_k, score_threshold)
        return finalize_contexts(
            question,
            chunks,
            index_path,
            allowed_document_ids=allowed_document_ids,
        )

    chunks = _search_keyword_chunks(
        question,
        top_k,
        index_path,
        score_threshold,
        allowed_document_ids,
    )
    return finalize_contexts(
        question,
        chunks,
        index_path,
        allowed_document_ids=allowed_document_ids,
    )


def build_fallback_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return (
            "I could not find relevant content in the current knowledge base. "
            "Please upload related documents first or ask a more specific question."
        )

    context_sections = []
    for context in contexts:
        context_sections.append(
            "\n".join(
                [
                    f"Source: {context.get('filename', 'unknown')}",
                    f"Chunk ID: {context.get('chunk_id', 'unknown')}",
                    context["content"].strip(),
                ]
            )
        )
    merged_context = "\n\n---\n\n".join(context_sections)
    return (
        "Based on the retrieved document context, here is the closest answer I can provide:\n\n"
        f"{merged_context}\n\n"
        f"Question: {question}"
    )


def build_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    *,
    security_mode: str = BASELINE_SECURITY_MODE,
) -> Dict[str, Any]:
    messages = build_rag_messages(
        question,
        contexts,
        security_mode=security_mode,
    )

    if not contexts:
        return {
            "answer": build_fallback_answer(question, contexts),
            "mode": "no_context",
            "model": None,
            "llm_error": None,
            "llm_error_code": None,
            "llm_latency_ms": None,
            "llm_usage": None,
            "citations": build_citations(contexts),
        }

    if not is_llm_configured():
        return {
            "answer": build_fallback_answer(question, contexts),
            "mode": "local_fallback",
            "model": None,
            "llm_error": "LLM API key is not configured.",
            "llm_error_code": "not_configured",
            "llm_latency_ms": None,
            "llm_usage": None,
            "citations": build_citations(contexts),
        }

    llm_started = perf_counter()
    try:
        completion = chat_completion(messages, include_metadata=True)
        llm_latency_ms = round((perf_counter() - llm_started) * 1000, 3)
        # A string remains accepted for existing injected fakes and callers.
        if isinstance(completion, ChatCompletionResult):
            answer = completion.answer
            model = completion.model
            usage = completion.token_usage()
        else:
            answer = completion
            model = settings.llm_model
            usage = None
        return {
            "answer": answer,
            "mode": "llm",
            "model": model,
            "llm_error": None,
            "llm_error_code": None,
            "llm_latency_ms": llm_latency_ms,
            "llm_usage": usage,
            "citations": build_citations(contexts),
        }
    except LLMClientError as exc:
        return {
            "answer": build_fallback_answer(question, contexts),
            "mode": "local_fallback",
            "model": settings.llm_model,
            "llm_error": exc.public_message,
            "llm_error_code": exc.code,
            "llm_latency_ms": round((perf_counter() - llm_started) * 1000, 3),
            "llm_usage": None,
            "citations": build_citations(contexts),
        }
