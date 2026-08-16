from pathlib import Path
from time import perf_counter
from typing import Any, Dict
from uuid import UUID

from app.core.config import settings
from app.observability.rag_logging import (
    build_rag_event,
    emit_rag_event,
    normalize_request_id,
)
from app.retrieval.reranker import RerankerModelError, SemanticReranker
from app.security.defenses import (
    SAFE_BLOCKED_RESPONSE,
    SecurityPolicy,
    analyze_context_security_signals,
    resolve_security_policy,
    validate_and_secure_output,
)
from app.services.embeddings import EmbeddingClientError
from app.services.access_control import get_readable_document_ids
from app.services.knowledge_base import build_answer, finalize_contexts, search_chunks
from app.services.prompts import protected_prompt_fragments
from app.services.search_service import (
    RerankedHybridConfig,
    run_reranked_hybrid_search,
)


def _safe_emit_rag_event(**event_fields: Any) -> Dict[str, Any] | None:
    """Keep diagnostics best-effort so a log failure cannot change RAG behavior."""
    try:
        event = build_rag_event(**event_fields)
        emit_rag_event(event)
        return event
    except Exception:
        return None


def _runtime_error_stage(exc: Exception, current_stage: str) -> str:
    if isinstance(exc, EmbeddingClientError):
        return "embedding"
    if isinstance(exc, RerankerModelError):
        return "reranking"
    return current_stage


def _secure_output_fail_closed(
    answer_result: Dict[str, Any],
    contexts: list[Dict[str, Any]],
    *,
    policy: SecurityPolicy,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        return validate_and_secure_output(
            answer_result,
            contexts,
            policy=policy,
            protected_prompt_fragments=protected_prompt_fragments(policy.mode),
        )
    except Exception:
        if not policy.layered:
            return answer_result, {
                "status": "not_applied",
                "blocked": False,
                "blocked_reason": None,
                "blocking_defense_id": None,
                "matched_prompt_fragment_count": 0,
                "protected_output_canary_matched": False,
            }
        secured = dict(answer_result)
        secured["answer"] = SAFE_BLOCKED_RESPONSE
        secured["citations"] = []
        return secured, {
            "status": "blocked",
            "blocked": True,
            "blocked_reason": "output_validator_error",
            "blocking_defense_id": "DEF-OUTPUT-001",
            "matched_prompt_fragment_count": 0,
            "protected_output_canary_matched": False,
        }


def resolve_min_score(retrieval_mode: str, requested_min_score: float | None) -> float:
    if requested_min_score is not None:
        return requested_min_score
    if retrieval_mode == "vector":
        return settings.vector_min_score
    if retrieval_mode == "hybrid":
        return settings.hybrid_min_score
    if retrieval_mode == "rerank":
        return settings.rerank_min_score
    if retrieval_mode == "hybrid_rerank":
        return 0.0
    return settings.keyword_min_score


def answer_question(
    question: str,
    top_k: int = 3,
    *,
    user_id: UUID,
    retrieval_mode: str = "keyword",
    min_score: float | None = None,
    index_path: Path | None = None,
    vector_index_path: Path | None = None,
    reranked_hybrid_config: RerankedHybridConfig | None = None,
    reranker: SemanticReranker | None = None,
    request_id: str | None = None,
    security_mode: str | None = None,
    protected_output_canaries: tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Run retrieval and answer generation as one reusable RAG operation."""
    resolved_request_id = normalize_request_id(request_id)
    policy = resolve_security_policy(
        security_mode or settings.rag_security_mode,
        protected_output_canaries=protected_output_canaries,
    )
    total_started = perf_counter()
    effective_min_score = resolve_min_score(retrieval_mode, min_score)
    current_stage = "authorization"
    retrieval_evidence: Dict[str, Any] | None = None
    contexts = []
    retrieved_results = []
    retrieval_component_ms: float | None = None
    rerank_ms: float | None = None
    context_build_ms: float | None = None
    generation_latency_ms: float | None = None
    security_signals = analyze_context_security_signals([], policy=policy)
    output_validation: Dict[str, Any] = {
        "status": "not_reached",
        "blocked": False,
        "blocked_reason": None,
        "blocking_defense_id": None,
        "matched_prompt_fragment_count": 0,
        "protected_output_canary_matched": False,
    }

    try:
        allowed_document_ids = get_readable_document_ids(user_id)
        current_stage = "retrieval"
        retrieval_started = perf_counter()
        if retrieval_mode == "hybrid_rerank":
            if reranked_hybrid_config is None:
                raise ValueError(
                    "hybrid_rerank requires an explicit RerankedHybridConfig"
                )
            if top_k != reranked_hybrid_config.final_top_k:
                raise ValueError(
                    "top_k must match reranked_hybrid_config.final_top_k"
                )
            retrieval_result = run_reranked_hybrid_search(
                question,
                configuration=reranked_hybrid_config,
                chunk_index_path=index_path,
                vector_index_path=vector_index_path,
                reranker=reranker,
                allowed_document_ids=allowed_document_ids,
            )
            retrieved_results = retrieval_result.results_after_rerank
            retrieval_component_ms = retrieval_result.retrieval_ms
            rerank_ms = retrieval_result.rerank_ms
            current_stage = "context"
            context_started = perf_counter()
            contexts = finalize_contexts(
                question,
                retrieved_results,
                index_path,
                allowed_document_ids=allowed_document_ids,
            )
            context_build_ms = round((perf_counter() - context_started) * 1000, 3)
            retrieval_evidence = {
                "configuration": retrieval_result.configuration.to_dict(),
                "candidates_before_rerank": retrieval_result.candidates_before_rerank,
                "results_after_rerank": retrieved_results,
            }
        else:
            contexts = search_chunks(
                question,
                top_k,
                index_path=index_path,
                retrieval_mode=retrieval_mode,
                vector_index_path=vector_index_path,
                min_score=effective_min_score,
                allowed_document_ids=allowed_document_ids,
            )
            retrieved_results = [
                context
                for context in contexts
                if context.get("context_role", "retrieved") == "retrieved"
            ]
        retrieval_latency_ms = round((perf_counter() - retrieval_started) * 1000, 3)
        if retrieval_component_ms is None:
            # Legacy search functions include any context finalization internally.
            retrieval_component_ms = retrieval_latency_ms

        security_signals = analyze_context_security_signals(contexts, policy=policy)
        current_stage = "llm"
        generation_started = perf_counter()
        answer_result = build_answer(
            question,
            contexts,
            security_mode=policy.mode,
        )
        generation_latency_ms = round((perf_counter() - generation_started) * 1000, 3)
        current_stage = "output_validation"
        answer_result, output_validation = _secure_output_fail_closed(
            answer_result,
            contexts,
            policy=policy,
        )
        total_latency_ms = round((perf_counter() - total_started) * 1000, 3)

        llm_error_code = answer_result.get("llm_error_code")
        event_status = "failed" if llm_error_code else "success"
        event = _safe_emit_rag_event(
            request_id=resolved_request_id,
            status=event_status,
            retrieval_mode=retrieval_mode,
            provider=settings.llm_provider,
            model=answer_result.get("model"),
            answer_mode=answer_result.get("mode"),
            query_length=len(question),
            retrieved_results=retrieved_results,
            contexts=contexts,
            citations=answer_result["citations"],
            retrieval_ms=retrieval_component_ms,
            rerank_ms=rerank_ms,
            context_build_ms=context_build_ms,
            generation_ms=generation_latency_ms,
            llm_ms=answer_result.get("llm_latency_ms"),
            total_ms=total_latency_ms,
            token_usage=answer_result.get("llm_usage"),
            error_stage="llm" if llm_error_code else None,
            error_type=llm_error_code,
            error_message=answer_result.get("llm_error") if llm_error_code else None,
            security_mode=policy.mode,
            security_policy_version=policy.version,
            enabled_defense_ids=policy.enabled_defense_ids,
            security_signals=security_signals,
            output_validation=output_validation,
        )

        return {
            "request_id": resolved_request_id,
            "question": question,
            "retrieval_mode": retrieval_mode,
            "min_score": effective_min_score,
            "answer": answer_result["answer"],
            "answer_mode": answer_result["mode"],
            "model": answer_result["model"],
            "llm_error": answer_result["llm_error"],
            "citations": answer_result["citations"],
            "contexts": contexts,
            "retrieval_evidence": retrieval_evidence,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_ms,
            "context_build_latency_ms": context_build_ms,
            "generation_latency_ms": generation_latency_ms,
            "llm_latency_ms": answer_result.get("llm_latency_ms"),
            "llm_usage": answer_result.get("llm_usage"),
            "total_latency_ms": total_latency_ms,
            "security": {
                "policy": policy.to_dict(),
                "context_signals": security_signals,
                "output_validation": output_validation,
            },
            "runtime_event": event,
        }
    except Exception as exc:
        total_latency_ms = round((perf_counter() - total_started) * 1000, 3)
        _safe_emit_rag_event(
            request_id=resolved_request_id,
            status="failed",
            retrieval_mode=retrieval_mode,
            provider=settings.llm_provider,
            model=settings.llm_model,
            answer_mode=None,
            query_length=len(question),
            retrieved_results=retrieved_results,
            contexts=contexts,
            citations=[],
            retrieval_ms=retrieval_component_ms,
            rerank_ms=rerank_ms,
            context_build_ms=context_build_ms,
            generation_ms=generation_latency_ms,
            llm_ms=None,
            total_ms=total_latency_ms,
            token_usage=None,
            error_stage=_runtime_error_stage(exc, current_stage),
            error_type=type(exc).__name__,
            error_message="RAG request failed before completion.",
            security_mode=policy.mode,
            security_policy_version=policy.version,
            enabled_defense_ids=policy.enabled_defense_ids,
            security_signals=security_signals,
            output_validation=output_validation,
        )
        raise
