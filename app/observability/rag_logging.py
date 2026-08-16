"""Safe, one-event-per-request JSONL logging for the production RAG path."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.core.config import BASE_DIR, settings


RAG_LOG_SCHEMA_VERSION = 2
_WRITE_LOCK = threading.Lock()
_FALLBACK_LOGGER = logging.getLogger(__name__)


class _RaisingFileHandler(logging.FileHandler):
    """Let the caller detect file-write failures that logging normally swallows."""

    def handleError(self, record: logging.LogRecord) -> None:
        raise


def new_request_id() -> str:
    """Create an opaque request identity without embedding query or user data."""
    return str(uuid4())


def normalize_request_id(request_id: str | None) -> str:
    """Create or validate the single UUID propagated through a RAG request."""
    if request_id is None:
        return new_request_id()
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty UUID string")
    try:
        return str(UUID(request_id.strip()))
    except ValueError as exc:
        raise ValueError("request_id must be a valid UUID string") from exc


def _optional_ms(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("latency values must be numeric or null")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("latency values must be finite and non-negative")
    return round(normalized, 3)


def _optional_token_count(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counts must be non-negative integers or null")
    return value


def _compact_retrieved_chunks(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    compact = []
    for result in results:
        chunk_id = result.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        item: dict[str, Any] = {
            "chunk_id": chunk_id,
            "document_id": (
                str(result["document_id"])
                if result.get("document_id") is not None
                else None
            ),
        }
        for field in (
            "score",
            "keyword_score",
            "vector_score",
            "fused_score",
            "pre_rerank_score",
            "rerank_score",
        ):
            value = result.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                item[field] = round(numeric, 6) if math.isfinite(numeric) else None
        source_ranks = result.get("source_ranks")
        if isinstance(source_ranks, Mapping):
            item["source_ranks"] = {
                str(source): int(rank)
                for source, rank in source_ranks.items()
                if isinstance(rank, int) and not isinstance(rank, bool)
            }
        source_scores = result.get("source_scores")
        if isinstance(source_scores, Mapping):
            item["source_scores"] = {
                str(source): round(float(score), 6)
                for source, score in source_scores.items()
                if isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            }
        compact.append(item)
    return compact


def _compact_identifiers(value: object, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of identifiers")
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 200:
            raise ValueError(f"{field} contains an invalid identifier")
        identifiers.append(item)
    return list(dict.fromkeys(identifiers))


def build_rag_event(
    *,
    request_id: str,
    status: str,
    retrieval_mode: str,
    provider: str,
    model: str | None,
    answer_mode: str | None,
    query_length: int,
    retrieved_results: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    citations: Sequence[Mapping[str, Any]],
    retrieval_ms: float | None,
    rerank_ms: float | None,
    context_build_ms: float | None,
    generation_ms: float | None,
    llm_ms: float | None,
    total_ms: float,
    token_usage: Mapping[str, Any] | None,
    error_stage: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    security_mode: str = "baseline",
    security_policy_version: str | None = None,
    enabled_defense_ids: Sequence[str] = (),
    security_signals: Mapping[str, Any] | None = None,
    output_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the content-minimized RAG event schema with security metadata."""
    normalized_request_id = normalize_request_id(request_id)
    if status not in {"success", "failed"}:
        raise ValueError("RAG event status must be success or failed")
    if not isinstance(query_length, int) or isinstance(query_length, bool) or query_length < 0:
        raise ValueError("query_length must be a non-negative integer")
    if security_mode not in {"baseline", "layered"}:
        raise ValueError("security_mode must be baseline or layered")

    compact_chunks = _compact_retrieved_chunks(retrieved_results)
    context_chunk_ids = [
        str(context["chunk_id"])
        for context in contexts
        if isinstance(context.get("chunk_id"), str) and context.get("chunk_id")
    ]
    cited_chunk_ids = [
        str(citation["chunk_id"])
        for citation in citations
        if isinstance(citation.get("chunk_id"), str) and citation.get("chunk_id")
    ]
    usage = token_usage or {}
    signals = security_signals or {}
    validation = output_validation or {}
    signal_ids = _compact_identifiers(
        signals.get("signal_ids"),
        field="security signal IDs",
    )
    flagged_chunk_ids = _compact_identifiers(
        signals.get("flagged_chunk_ids"),
        field="security signal chunk IDs",
    )
    defense_ids = _compact_identifiers(
        enabled_defense_ids,
        field="enabled defense IDs",
    )
    event = {
        "schema_version": RAG_LOG_SCHEMA_VERSION,
        "event": "rag_request_completed" if status == "success" else "rag_request_failed",
        "request_id": normalized_request_id,
        "status": status,
        "retrieval_mode": retrieval_mode,
        "provider": provider,
        "model": model,
        "answer_mode": answer_mode,
        "llm_called": answer_mode == "llm" or (
            answer_mode == "local_fallback" and model is not None
        ),
        "query_length": query_length,
        "retrieved_chunk_ids": [item["chunk_id"] for item in compact_chunks],
        "retrieved_document_ids": list(
            dict.fromkeys(
                item["document_id"]
                for item in compact_chunks
                if item.get("document_id") is not None
            )
        ),
        "retrieved_chunks": compact_chunks,
        "context_chunk_ids": context_chunk_ids,
        "citation_count": len(cited_chunk_ids),
        "cited_chunk_ids": cited_chunk_ids,
        "embedding_ms": None,
        "retrieval_ms": _optional_ms(retrieval_ms),
        "rerank_ms": _optional_ms(rerank_ms),
        "context_build_ms": _optional_ms(context_build_ms),
        "generation_ms": _optional_ms(generation_ms),
        "llm_ms": _optional_ms(llm_ms),
        "total_ms": _optional_ms(total_ms),
        "prompt_tokens": _optional_token_count(usage.get("prompt_tokens")),
        "completion_tokens": _optional_token_count(usage.get("completion_tokens")),
        "total_tokens": _optional_token_count(usage.get("total_tokens")),
        "error_stage": error_stage,
        "error_type": error_type,
        "error_message": error_message,
        "security_mode": security_mode,
        "security_policy_version": security_policy_version,
        "enabled_defense_ids": defense_ids,
        "security_signal_status": signals.get("status", "not_applied"),
        "security_signal_action": signals.get("action", "none"),
        "security_signal_ids": signal_ids,
        "security_signal_count": len(flagged_chunk_ids),
        "security_signal_chunk_ids": flagged_chunk_ids,
        "output_validation_status": validation.get("status", "not_reached"),
        "output_blocked": bool(validation.get("blocked", False)),
        "blocked_reason": validation.get("blocked_reason"),
        "blocking_defense_id": validation.get("blocking_defense_id"),
    }
    serialize_rag_event(event)
    return event


def serialize_rag_event(event: Mapping[str, Any]) -> str:
    """Return one compact JSON line and reject non-serializable fields."""
    return json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"))


def _resolved_log_path() -> Path:
    configured = Path(settings.rag_structured_log_path)
    return configured if configured.is_absolute() else BASE_DIR / configured


def _write_json_line(payload: str, *, level: int) -> None:
    path = _resolved_log_path()
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = _RaisingFileHandler(path, mode="a", encoding="utf-8")
        try:
            handler.setFormatter(logging.Formatter("%(message)s"))
            record = logging.LogRecord(
                name="enterprise_rag.runtime",
                level=level,
                pathname=__file__,
                lineno=0,
                msg=payload,
                args=(),
                exc_info=None,
            )
            handler.emit(record)
        finally:
            handler.close()


def emit_rag_event(event: Mapping[str, Any]) -> bool:
    """Best-effort emission: logger failures never replace the business result."""
    if not settings.rag_structured_logging_enabled:
        return False
    try:
        payload = serialize_rag_event(event)
        level = logging.INFO if event.get("status") == "success" else logging.ERROR
        _write_json_line(payload, level=level)
        return True
    except Exception as exc:
        try:
            _FALLBACK_LOGGER.error(
                "Structured RAG log emission failed for request_id=%s error_type=%s",
                event.get("request_id"),
                type(exc).__name__,
            )
        except Exception:
            pass
        return False
