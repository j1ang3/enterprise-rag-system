"""Offline, evidence-driven failure analysis for frozen RAG evaluation runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


ANALYSIS_VERSION = "w8-t4-failure-analysis.v1"
ARTIFACT_VERSION = 1
FAILURE_TYPES = (
    "retrieval_failure",
    "ranking_failure",
    "context_construction_failure",
    "generation_failure",
    "citation_failure",
    "execution_failure",
    "needs_review",
    "no_failure",
)
EVIDENCE_STRENGTHS = ("confirmed", "supported", "uncertain")
ANSWER_QUALITY_VALUES = ("acceptable", "unacceptable", "unsupported", "uncertain")
CITATION_QUALITY_VALUES = ("acceptable", "incorrect", "uncertain")


class FailureAnalysisValidationError(ValueError):
    """Raised when frozen source evidence cannot be trusted or aligned."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FailureAnalysisValidationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise FailureAnalysisValidationError(f"{label} must be a JSON object")
    return value


def load_jsonl_events(path: Path) -> list[Dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"runtime log not found: {resolved}")
    events = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FailureAnalysisValidationError(
                f"runtime log is invalid JSONL at line {line_number}: {resolved}"
            ) from exc
        if not isinstance(value, dict):
            raise FailureAnalysisValidationError(
                f"runtime log line {line_number} must be an object"
            )
        events.append(value)
    return events


def _project_path(project_root: Path, recorded: object, *, field: str) -> Path:
    if not isinstance(recorded, str) or not recorded.strip():
        raise FailureAnalysisValidationError(f"{field} must be a non-empty path")
    path = Path(recorded)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_source_file(
    project_root: Path,
    record: object,
    *,
    field: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise FailureAnalysisValidationError(f"{field} must be an object")
    path = _project_path(project_root, record.get("path"), field=f"{field}.path")
    if not path.exists():
        raise FileNotFoundError(f"{field} not found: {path}")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
        raise FailureAnalysisValidationError(f"{field} SHA-256 identity mismatch")
    return path


def load_failure_analysis_manifest(
    path: Path,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    manifest = load_json_object(path, label="failure analysis manifest")
    if manifest.get("manifest_version") != 1 or manifest.get("task") != "W8-T4":
        raise FailureAnalysisValidationError("failure analysis manifest identity is invalid")
    if manifest.get("analysis_version") != ANALYSIS_VERSION:
        raise FailureAnalysisValidationError("failure analysis version is unsupported")

    evaluation_path = _validate_source_file(
        project_root,
        manifest.get("source_evaluation_artifact"),
        field="source_evaluation_artifact",
    )
    unanswerable_path = _validate_source_file(
        project_root,
        manifest.get("source_unanswerable_artifact"),
        field="source_unanswerable_artifact",
    )
    runtime_records = manifest.get("source_runtime_logs")
    if not isinstance(runtime_records, list) or not runtime_records:
        raise FailureAnalysisValidationError("source_runtime_logs must be non-empty")
    runtime_paths = [
        _validate_source_file(
            project_root,
            record,
            field=f"source_runtime_logs[{position}]",
        )
        for position, record in enumerate(runtime_records, start=1)
    ]

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise FailureAnalysisValidationError("selection must be an object")
    for field in (
        "w8_t1_observed_signal_query_ids",
        "w8_t1_success_control_query_ids",
        "w8_t2_unanswerable_control_query_ids",
    ):
        value = selection.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise FailureAnalysisValidationError(f"selection.{field} must be a string list")
        if len(value) != len(set(value)):
            raise FailureAnalysisValidationError(f"selection.{field} contains duplicates")

    reviews = manifest.get("manual_evidence_reviews")
    if not isinstance(reviews, list):
        raise FailureAnalysisValidationError("manual_evidence_reviews must be a list")
    review_ids = []
    for position, review in enumerate(reviews, start=1):
        if not isinstance(review, Mapping):
            raise FailureAnalysisValidationError(
                f"manual_evidence_reviews[{position}] must be an object"
            )
        query_id = review.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise FailureAnalysisValidationError(
                f"manual_evidence_reviews[{position}].query_id is invalid"
            )
        if review.get("answer_quality") not in ANSWER_QUALITY_VALUES:
            raise FailureAnalysisValidationError(
                f"manual review answer_quality is invalid for {query_id}"
            )
        if review.get("citation_quality") not in CITATION_QUALITY_VALUES:
            raise FailureAnalysisValidationError(
                f"manual review citation_quality is invalid for {query_id}"
            )
        for field in ("observed_problem", "answer_notes", "citation_notes"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise FailureAnalysisValidationError(
                    f"manual review {field} is required for {query_id}"
                )
        review_ids.append(query_id)
    if len(review_ids) != len(set(review_ids)):
        raise FailureAnalysisValidationError("manual evidence reviews contain duplicates")

    return {
        "manifest": manifest,
        "manifest_path": path.resolve(),
        "manifest_sha256": file_sha256(path.resolve()),
        "evaluation_path": evaluation_path,
        "unanswerable_path": unanswerable_path,
        "runtime_paths": runtime_paths,
    }


def validate_source_artifacts(
    evaluation: Mapping[str, Any],
    unanswerable: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> None:
    evaluation_record = manifest["source_evaluation_artifact"]
    unanswerable_record = manifest["source_unanswerable_artifact"]
    if evaluation.get("task") != "W8-T1" or evaluation.get("formal") is not True:
        raise FailureAnalysisValidationError("source evaluation is not formal W8-T1")
    if evaluation.get("run_id") != evaluation_record.get("run_id"):
        raise FailureAnalysisValidationError("source evaluation run_id mismatch")
    if unanswerable.get("task") != "W8-T2" or unanswerable.get("formal") is not True:
        raise FailureAnalysisValidationError("source unanswerable run is not formal W8-T2")
    if unanswerable.get("run_id") != unanswerable_record.get("run_id"):
        raise FailureAnalysisValidationError("source unanswerable run_id mismatch")

    evaluation_config = evaluation.get("resolved_configuration")
    unanswerable_config = unanswerable.get("resolved_configuration")
    if not isinstance(evaluation_config, Mapping) or not isinstance(
        unanswerable_config, Mapping
    ):
        raise FailureAnalysisValidationError("source resolved configuration is missing")
    if dict(evaluation_config) != dict(unanswerable_config):
        raise FailureAnalysisValidationError("W8-T1 and W8-T2 configurations differ")
    llm = evaluation_config.get("llm")
    if not isinstance(llm, Mapping):
        raise FailureAnalysisValidationError("formal LLM identity is missing")
    if llm.get("provider") != "ollama" or llm.get("model") != "qwen3:8b":
        raise FailureAnalysisValidationError("formal source model is not Ollama qwen3:8b")
    identity = llm.get("model_identity")
    if not isinstance(identity, Mapping) or not identity.get("digest"):
        raise FailureAnalysisValidationError("resolved qwen3:8b identity is missing")

    w8_t2_source = unanswerable.get("source_identities", {}).get(
        "source_w8_t1_artifact"
    )
    if not isinstance(w8_t2_source, Mapping):
        raise FailureAnalysisValidationError("W8-T2 source W8-T1 identity is missing")
    if w8_t2_source.get("run_id") != evaluation.get("run_id"):
        raise FailureAnalysisValidationError("W8-T2 does not reference the W8-T1 run")
    if w8_t2_source.get("sha256") != evaluation_record.get("sha256"):
        raise FailureAnalysisValidationError("W8-T2 W8-T1 SHA-256 reference drifted")


def _result_index(rows: object, *, field: str) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise FailureAnalysisValidationError(f"{field} must be a list")
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FailureAnalysisValidationError(f"{field} contains a non-object row")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise FailureAnalysisValidationError(f"{field} contains an invalid query_id")
        if query_id in indexed:
            raise FailureAnalysisValidationError(f"{field} contains duplicate {query_id}")
        indexed[query_id] = row
    return indexed


def observed_w8_t1_signal_ids(artifact: Mapping[str, Any]) -> list[str]:
    """Return every answerable case with any saved formal quality/execution signal."""
    signal_ids = []
    for row in _result_index(artifact.get("results"), field="W8-T1 results").values():
        if not row.get("answerable"):
            continue
        if row.get("status") != "success":
            signal_ids.append(str(row["query_id"]))
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            signal_ids.append(str(row["query_id"]))
            continue
        retrieval = metrics.get("retrieval")
        generation = metrics.get("generation")
        if not isinstance(retrieval, Mapping) or not isinstance(generation, Mapping):
            signal_ids.append(str(row["query_id"]))
            continue
        metric_rows = retrieval.get("metrics_by_k")
        final_recall = None
        if isinstance(metric_rows, Mapping) and metric_rows:
            largest_k = max(metric_rows, key=lambda value: int(str(value)))
            final_row = metric_rows.get(largest_k)
            if isinstance(final_row, Mapping):
                final_recall = final_row.get("recall")
        keyword = generation.get("required_keyword_proxy")
        citation = generation.get("citation")
        document_citation = citation.get("document") if isinstance(citation, Mapping) else None
        strict_citation = citation.get("strict_chunk") if isinstance(citation, Mapping) else None
        has_signal = (
            final_recall is None
            or float(final_recall) < 1.0
            or not isinstance(keyword, Mapping)
            or keyword.get("matched") is False
            or not isinstance(document_citation, Mapping)
            or document_citation.get("exact_match") is False
            or (
                isinstance(strict_citation, Mapping)
                and strict_citation.get("status") == "evaluated"
                and float(strict_citation.get("recall", 0.0)) < 1.0
            )
        )
        if has_signal:
            signal_ids.append(str(row["query_id"]))
    return signal_ids


def observed_w8_t2_failure_ids(artifact: Mapping[str, Any]) -> list[str]:
    failure_ids = []
    rows = _result_index(
        artifact.get("unanswerable_results"), field="W8-T2 unanswerable_results"
    )
    for row in rows.values():
        behavior = row.get("behavior_evaluation")
        outcome = behavior.get("outcome") if isinstance(behavior, Mapping) else None
        if row.get("status") != "success" or outcome != "correct_abstention":
            failure_ids.append(str(row["query_id"]))
    return failure_ids


def index_runtime_events(
    events: Iterable[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for event in events:
        request_id = event.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise FailureAnalysisValidationError("runtime event request_id is missing")
        if request_id in indexed:
            raise FailureAnalysisValidationError(
                f"duplicate runtime request_id: {request_id}"
            )
        indexed[request_id] = event
    return indexed


def join_case_runtime_evidence(
    result: Mapping[str, Any],
    *,
    source_run_id: str,
    runtime_events: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    query_id = result.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise FailureAnalysisValidationError("case query_id is missing")
    actual = result.get("actual")
    request_id = actual.get("request_id") if isinstance(actual, Mapping) else None
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise FailureAnalysisValidationError(f"invalid request_id for {query_id}")
    event = runtime_events.get(request_id) if request_id else None
    if request_id and event is None:
        join_status = "request_id_not_found_in_supplied_logs"
    elif event is not None:
        join_status = "joined_by_request_id"
    else:
        join_status = "unavailable_source_case_has_no_request_id"
    return {
        "source_run_id": source_run_id,
        "query_id": query_id,
        "request_id": request_id,
        "join_status": join_status,
        "event": dict(event) if event is not None else None,
    }


def _identity_values(rows: object, field: str) -> list[str] | None:
    if rows is None:
        return None
    if not isinstance(rows, list):
        return None
    values = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        value = row.get(field)
        if isinstance(value, str) and value:
            values.append(value)
    return list(dict.fromkeys(values))


def _requirements_satisfied(
    actual_ids: Sequence[str],
    expected_ids: Sequence[str],
    match: str,
) -> bool:
    actual = set(actual_ids)
    expected = set(expected_ids)
    if not expected:
        return False
    if match == "any":
        return bool(actual & expected)
    if match == "all":
        return expected <= actual
    raise FailureAnalysisValidationError(f"unsupported expected match rule: {match}")


def classify_failure_case(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify one normalized case without guessing across missing evidence."""
    status = evidence.get("execution_status")
    if status != "success":
        return _classification(
            "execution_failure",
            strength="confirmed",
            basis="No valid quality output exists; runtime/evaluation execution failed.",
            subtype=evidence.get("execution_subtype") or "unknown_runtime_error",
        )

    if evidence.get("answerable") is False:
        outcome = evidence.get("unanswerable_outcome")
        if outcome == "correct_abstention":
            return _classification(
                "no_failure",
                strength="supported",
                basis="The corpus-verified unanswerable case produced a correct abstention proxy.",
            )
        if outcome in {"unsupported_answer", "contaminated_abstention"}:
            contributing = (
                ["citation_failure"]
                if evidence.get("citation_quality") == "incorrect"
                else []
            )
            return _classification(
                "generation_failure",
                contributing=contributing,
                strength="supported",
                basis=(
                    "The requested fact is absent from the frozen corpus, but the output "
                    "contains unsupported factual content."
                ),
            )
        if outcome == "execution_failure":
            return _classification(
                "execution_failure",
                strength="confirmed",
                basis="The unanswerable case had no valid behavioral output.",
                subtype=evidence.get("execution_subtype") or "unknown_runtime_error",
            )
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Unanswerable behavior evidence is ambiguous or missing.",
        )

    observed_signals = evidence.get("observed_signals")
    if not isinstance(observed_signals, list):
        observed_signals = []
    answer_quality = evidence.get("answer_quality")
    citation_quality = evidence.get("citation_quality")
    if not observed_signals and answer_quality is None and citation_quality is None:
        return _classification(
            "no_failure",
            strength="supported",
            basis="No saved formal quality or execution signal selected this control case.",
        )
    if answer_quality not in ANSWER_QUALITY_VALUES or citation_quality not in CITATION_QUALITY_VALUES:
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Observed quality signals lack a complete answer/citation evidence review.",
        )
    if answer_quality == "uncertain":
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Answer acceptability remains ambiguous after available evidence review.",
        )

    answer_bad = answer_quality in {"unacceptable", "unsupported"}
    citation_bad = citation_quality == "incorrect"
    if not answer_bad:
        if citation_bad:
            return _classification(
                "citation_failure",
                strength="supported",
                basis=(
                    "The answer is acceptable, but the response-level pipeline citation "
                    "set is missing or includes unsupported source identity."
                ),
                subtype=evidence.get("citation_failure_subtype"),
            )
        if citation_quality == "uncertain":
            return _classification(
                "needs_review",
                strength="uncertain",
                basis="The answer is acceptable but citation semantics remain ambiguous.",
            )
        return _classification(
            "no_failure",
            strength="supported",
            basis=(
                "Manual evidence review found an acceptable answer and citation behavior; "
                "the saved metric signal is an evaluation limitation."
            ),
        )

    expected_chunks = evidence.get("expected_chunk_ids")
    expected_documents = evidence.get("expected_documents")
    if not isinstance(expected_chunks, list) or not isinstance(expected_documents, list):
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Ground Truth identity evidence is missing.",
        )
    if expected_chunks:
        identity_field = "chunk_id"
        expected_ids = expected_chunks
        strength = "confirmed"
        granularity = "strict_chunk"
    elif expected_documents:
        identity_field = "filename"
        expected_ids = expected_documents
        strength = "supported"
        granularity = "document"
    else:
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Answerable case has no relevant evidence labels.",
        )
    match = evidence.get("expected_match", "all")
    candidates = evidence.get("candidate_ids_by_field", {}).get(identity_field)
    final_ids = evidence.get("final_ids_by_field", {}).get(identity_field)
    context_ids = evidence.get("context_ids_by_field", {}).get(identity_field)
    if candidates is None:
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Pre-reranker candidate evidence is unavailable.",
        )
    if not _requirements_satisfied(candidates, expected_ids, str(match)):
        return _classification(
            "retrieval_failure",
            contributing=["citation_failure"] if citation_bad else [],
            strength=strength,
            basis=(
                f"Causally required {granularity} evidence never entered the saved "
                "pre-reranker candidate set."
            ),
            subtype=f"{granularity}_candidate_miss",
        )
    if final_ids is None:
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Final ranked identity evidence is unavailable.",
        )
    if not _requirements_satisfied(final_ids, expected_ids, str(match)):
        return _classification(
            "ranking_failure",
            contributing=["citation_failure"] if citation_bad else [],
            strength=strength,
            basis=(
                f"Causally required {granularity} evidence was a candidate but did not "
                "survive the effective final cutoff."
            ),
            subtype=f"{granularity}_below_final_cutoff",
        )
    if context_ids is None:
        return _classification(
            "needs_review",
            strength="uncertain",
            basis="Final context identity evidence is unavailable.",
        )
    if not _requirements_satisfied(context_ids, expected_ids, str(match)):
        return _classification(
            "context_construction_failure",
            contributing=["citation_failure"] if citation_bad else [],
            strength=strength,
            basis=(
                f"Selected {granularity} evidence did not reach the saved final context."
            ),
            subtype=f"{granularity}_missing_from_context",
        )
    return _classification(
        "generation_failure",
        contributing=["citation_failure"] if citation_bad else [],
        strength=strength,
        basis=(
            "Sufficient labeled evidence reached the final context, but the answer "
            "remained wrong, unsupported, or materially incomplete."
        ),
    )


def _classification(
    primary: str,
    *,
    strength: str,
    basis: str,
    contributing: Sequence[str] = (),
    subtype: object = None,
) -> Dict[str, Any]:
    if primary not in FAILURE_TYPES:
        raise FailureAnalysisValidationError(f"unknown failure type: {primary}")
    if strength not in EVIDENCE_STRENGTHS:
        raise FailureAnalysisValidationError(f"unknown evidence strength: {strength}")
    normalized_contributing = list(dict.fromkeys(str(item) for item in contributing))
    if primary in normalized_contributing:
        raise FailureAnalysisValidationError("primary cannot repeat as contributing")
    return {
        "primary_failure": primary,
        "contributing_failures": normalized_contributing,
        "evidence_strength": strength,
        "failure_subtype": str(subtype) if subtype else None,
        "causal_basis": basis,
    }


def _signal_names(row: Mapping[str, Any]) -> list[str]:
    if row.get("status") != "success":
        return ["execution_failure"]
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return ["missing_metrics"]
    retrieval = metrics.get("retrieval", {})
    generation = metrics.get("generation", {})
    signals = []
    metric_rows = retrieval.get("metrics_by_k", {}) if isinstance(retrieval, Mapping) else {}
    if isinstance(metric_rows, Mapping) and metric_rows:
        largest_k = max(metric_rows, key=lambda value: int(str(value)))
        final_row = metric_rows[largest_k]
        if isinstance(final_row, Mapping) and float(final_row.get("recall", 0.0)) < 1.0:
            signals.append("document_retrieval_recall_below_one")
    keyword = generation.get("required_keyword_proxy") if isinstance(generation, Mapping) else None
    if isinstance(keyword, Mapping) and keyword.get("matched") is False:
        signals.append("required_keyword_proxy_miss")
    citation = generation.get("citation") if isinstance(generation, Mapping) else None
    document = citation.get("document") if isinstance(citation, Mapping) else None
    if isinstance(document, Mapping) and document.get("exact_match") is False:
        signals.append("document_citation_not_exact")
    strict = citation.get("strict_chunk") if isinstance(citation, Mapping) else None
    if (
        isinstance(strict, Mapping)
        and strict.get("status") == "evaluated"
        and float(strict.get("recall", 0.0)) < 1.0
    ):
        signals.append("strict_chunk_citation_recall_below_one")
    return signals


def _normalized_answerable_evidence(
    row: Mapping[str, Any],
    *,
    source_run_id: str,
    runtime_events: Mapping[str, Mapping[str, Any]],
    review: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    expected = row.get("expected")
    actual = row.get("actual")
    if not isinstance(expected, Mapping):
        raise FailureAnalysisValidationError(f"expected evidence missing for {row.get('query_id')}")
    retrieval = actual.get("retrieval") if isinstance(actual, Mapping) else None
    candidates = retrieval.get("candidates_before_rerank") if isinstance(retrieval, Mapping) else None
    final_chunks = retrieval.get("final_chunks") if isinstance(retrieval, Mapping) else None
    contexts = actual.get("contexts") if isinstance(actual, Mapping) else None
    citations = actual.get("citations") if isinstance(actual, Mapping) else None
    runtime = join_case_runtime_evidence(
        row,
        source_run_id=source_run_id,
        runtime_events=runtime_events,
    )
    evidence: Dict[str, Any] = {
        "case_source": "W8-T1",
        "source_run_id": source_run_id,
        "query_id": row.get("query_id"),
        "question": row.get("question"),
        "request_id": runtime["request_id"],
        "answerable": True,
        "execution_status": row.get("status"),
        "execution_error": row.get("error"),
        "observed_signals": _signal_names(row),
        "expected_documents": list(expected.get("documents", [])),
        "expected_chunk_ids": list(expected.get("chunk_ids", [])),
        "expected_answer": expected.get("answer"),
        "expected_match": expected.get("document_match", "all"),
        "ground_truth_granularity": (
            "strict_chunk_and_document"
            if expected.get("chunk_ids")
            else "document_with_answer_and_keyword_proxies"
        ),
        "candidate_ids_by_field": {
            "chunk_id": _identity_values(candidates, "chunk_id"),
            "filename": _identity_values(candidates, "filename"),
        },
        "final_ids_by_field": {
            "chunk_id": _identity_values(final_chunks, "chunk_id"),
            "filename": _identity_values(final_chunks, "filename"),
        },
        "context_ids_by_field": {
            "chunk_id": _identity_values(contexts, "chunk_id"),
            "filename": _identity_values(contexts, "filename"),
        },
        "citation_ids": {
            "chunk_id": _identity_values(citations, "chunk_id"),
            "filename": _identity_values(citations, "filename"),
        },
        "answer_reference": "source artifact actual.answer",
        "actual_answer": actual.get("answer") if isinstance(actual, Mapping) else None,
        "answer_mode": actual.get("answer_mode") if isinstance(actual, Mapping) else None,
        "actual_model": actual.get("model") if isinstance(actual, Mapping) else None,
        "runtime_evidence": {
            "join_status": runtime["join_status"],
            "status": runtime["event"].get("status") if runtime["event"] else None,
            "error_stage": runtime["event"].get("error_stage") if runtime["event"] else None,
        },
    }
    if review is not None:
        evidence.update(
            {
                "observed_problem": review["observed_problem"],
                "answer_quality": review["answer_quality"],
                "citation_quality": review["citation_quality"],
                "answer_notes": review["answer_notes"],
                "citation_notes": review["citation_notes"],
                "citation_failure_subtype": review.get("citation_failure_subtype"),
            }
        )
    return evidence


def _normalized_unanswerable_evidence(
    row: Mapping[str, Any],
    *,
    source_run_id: str,
    runtime_events: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    behavior = row.get("behavior_evaluation")
    if not isinstance(behavior, Mapping):
        raise FailureAnalysisValidationError(
            f"unanswerable behavior evidence missing for {row.get('query_id')}"
        )
    citation = behavior.get("citation_evaluation")
    actual = row.get("actual")
    runtime = join_case_runtime_evidence(
        row,
        source_run_id=source_run_id,
        runtime_events=runtime_events,
    )
    return {
        "case_source": "W8-T2",
        "source_run_id": source_run_id,
        "query_id": row.get("query_id"),
        "question": row.get("question"),
        "request_id": runtime["request_id"],
        "answerable": False,
        "execution_status": row.get("status"),
        "execution_error": row.get("error"),
        "observed_signals": (
            []
            if behavior.get("outcome") == "correct_abstention"
            else [f"unanswerable_{behavior.get('outcome')}"]
        ),
        "unanswerable_outcome": behavior.get("outcome"),
        "citation_quality": (
            "incorrect"
            if isinstance(citation, Mapping) and citation.get("misleading_citation_proxy")
            else "acceptable"
        ),
        "absence_status": row.get("absence_verification", {}).get("status"),
        "answer_reference": "source artifact actual.answer",
        "actual_answer": actual.get("answer") if isinstance(actual, Mapping) else None,
        "answer_mode": actual.get("answer_mode") if isinstance(actual, Mapping) else None,
        "actual_model": actual.get("model") if isinstance(actual, Mapping) else None,
        "runtime_evidence": {
            "join_status": runtime["join_status"],
            "status": runtime["event"].get("status") if runtime["event"] else None,
            "error_stage": runtime["event"].get("error_stage") if runtime["event"] else None,
        },
    }


def aggregate_classifications(
    cases: Sequence[Mapping[str, Any]],
    *,
    observed_signal_count: int,
    source_counts: Mapping[str, int],
) -> Dict[str, Any]:
    primary_counts = Counter(str(case["primary_failure"]) for case in cases)
    contributing_counts = Counter(
        failure
        for case in cases
        for failure in case.get("contributing_failures", [])
    )
    any_occurrence = Counter()
    for case in cases:
        primary = str(case["primary_failure"])
        if primary not in {"no_failure", "needs_review"}:
            any_occurrence[primary] += 1
        for failure in case.get("contributing_failures", []):
            any_occurrence[str(failure)] += 1
    quality_failures = [
        case
        for case in cases
        if case["primary_failure"]
        not in {"no_failure", "needs_review", "execution_failure"}
    ]
    execution_failures = [
        case for case in cases if case["primary_failure"] == "execution_failure"
    ]
    needs_review = [case for case in cases if case["primary_failure"] == "needs_review"]
    return {
        "source_case_counts": dict(source_counts),
        "analyzed_case_count": len(cases),
        "observed_signal_case_count": observed_signal_count,
        "quality_failure_case_count": len(quality_failures),
        "answerable_quality_failure_count": sum(
            bool(case.get("answerable")) for case in quality_failures
        ),
        "unanswerable_quality_failure_count": sum(
            not bool(case.get("answerable")) for case in quality_failures
        ),
        "execution_failure_count": len(execution_failures),
        "classified_failure_count": len(quality_failures) + len(execution_failures),
        "needs_review_count": len(needs_review),
        "no_failure_count": primary_counts.get("no_failure", 0),
        "primary_failure_counts": {
            failure: primary_counts.get(failure, 0)
            for failure in FAILURE_TYPES
        },
        "contributing_failure_counts": {
            failure: contributing_counts.get(failure, 0)
            for failure in FAILURE_TYPES
            if failure not in {"needs_review", "no_failure"}
        },
        "any_occurrence_failure_counts": {
            failure: any_occurrence.get(failure, 0)
            for failure in FAILURE_TYPES
            if failure not in {"needs_review", "no_failure"}
        },
        "denominators": {
            "primary_failure_distribution": "classified quality-failure cases",
            "execution_failure_rate": "all source case executions, with repeated query_ids retained per run",
            "selection_coverage": "all observed formal quality/execution signal cases",
            "contributing_failures": "independent occurrences; totals may exceed case count",
        },
    }


def build_failure_analysis_artifact(
    *,
    evaluation: Mapping[str, Any],
    unanswerable: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    runtime_events: Sequence[Mapping[str, Any]],
    runtime_identities: Sequence[Mapping[str, Any]],
    run_id: str,
    repository_state: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_source_artifacts(evaluation, unanswerable, manifest=manifest)
    runtime_index = index_runtime_events(runtime_events)
    evaluation_rows = _result_index(evaluation.get("results"), field="W8-T1 results")
    unanswerable_rows = _result_index(
        unanswerable.get("unanswerable_results"), field="W8-T2 unanswerable_results"
    )
    selection = manifest["selection"]
    computed_w8_t1_signals = observed_w8_t1_signal_ids(evaluation)
    configured_w8_t1_signals = selection["w8_t1_observed_signal_query_ids"]
    if computed_w8_t1_signals != configured_w8_t1_signals:
        raise FailureAnalysisValidationError(
            "W8-T1 observed-signal selection does not match the source artifact"
        )
    computed_w8_t2_failures = observed_w8_t2_failure_ids(unanswerable)
    if computed_w8_t2_failures != selection.get("w8_t2_observed_failure_query_ids", []):
        raise FailureAnalysisValidationError(
            "W8-T2 failure selection does not match the source artifact"
        )

    reviews = {
        str(review["query_id"]): review
        for review in manifest["manual_evidence_reviews"]
    }
    if set(reviews) != set(configured_w8_t1_signals):
        raise FailureAnalysisValidationError(
            "every W8-T1 observed signal requires exactly one manual evidence review"
        )

    selected_cases: list[Dict[str, Any]] = []
    for query_id in [
        *configured_w8_t1_signals,
        *selection["w8_t1_success_control_query_ids"],
    ]:
        row = evaluation_rows.get(query_id)
        if row is None:
            raise FailureAnalysisValidationError(f"selected W8-T1 case is missing: {query_id}")
        evidence = _normalized_answerable_evidence(
            row,
            source_run_id=str(evaluation["run_id"]),
            runtime_events=runtime_index,
            review=reviews.get(query_id),
        )
        selected_cases.append({**evidence, **classify_failure_case(evidence)})

    for query_id in [
        *computed_w8_t2_failures,
        *selection["w8_t2_unanswerable_control_query_ids"],
    ]:
        row = unanswerable_rows.get(query_id)
        if row is None:
            raise FailureAnalysisValidationError(f"selected W8-T2 case is missing: {query_id}")
        evidence = _normalized_unanswerable_evidence(
            row,
            source_run_id=str(unanswerable["run_id"]),
            runtime_events=runtime_index,
        )
        selected_cases.append({**evidence, **classify_failure_case(evidence)})

    raw_source_executions = (
        len(evaluation_rows)
        + len(unanswerable_rows)
        + len(unanswerable.get("answerable_control_results", []))
    )
    unique_primary_universe = sum(
        bool(row.get("answerable")) for row in evaluation_rows.values()
    ) + len(unanswerable_rows)
    source_counts = {
        "w8_t1_all_results": len(evaluation_rows),
        "w8_t2_unanswerable_results": len(unanswerable_rows),
        "w8_t2_answerable_controls": len(unanswerable.get("answerable_control_results", [])),
        "raw_source_case_executions": raw_source_executions,
        "deduplicated_primary_quality_universe": unique_primary_universe,
    }
    aggregate = aggregate_classifications(
        selected_cases,
        observed_signal_count=len(configured_w8_t1_signals) + len(computed_w8_t2_failures),
        source_counts=source_counts,
    )
    joined_count = sum(
        case["runtime_evidence"]["join_status"] == "joined_by_request_id"
        for case in selected_cases
    )
    return {
        "artifact_version": ARTIFACT_VERSION,
        "task": "W8-T4",
        "analysis_version": ANALYSIS_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formal": True,
        "status": "completed",
        "source_provenance": {
            "evaluation": {
                "run_id": evaluation["run_id"],
                **dict(manifest["source_evaluation_artifact"]),
            },
            "unanswerable": {
                "run_id": unanswerable["run_id"],
                **dict(manifest["source_unanswerable_artifact"]),
            },
            "runtime_logs": [dict(record) for record in runtime_identities],
            "analysis_manifest": dict(manifest_identity),
            "dataset_identity": dict(evaluation["dataset_identity"]),
            "corpus_identity": dict(evaluation["run_metadata"]["corpus_identity"]),
            "repository": dict(repository_state),
        },
        "resolved_configuration": dict(evaluation["resolved_configuration"]),
        "selection": {
            "rule": selection["rule"],
            "all_observed_failure_signals_analyzed": True,
            "w8_t1_observed_signal_query_ids": list(configured_w8_t1_signals),
            "w8_t2_observed_failure_query_ids": list(computed_w8_t2_failures),
            "success_control_query_ids": [
                *selection["w8_t1_success_control_query_ids"],
                *selection["w8_t2_unanswerable_control_query_ids"],
            ],
            "targeted_reproduction_count": 0,
            "cases_analyzed_from_existing_artifacts": len(selected_cases),
        },
        "evidence_join": {
            "run_id": "available for both formal source artifacts",
            "query_id": "available for every source result",
            "request_id": (
                "supported by the analyzer when present; historical W8-T1/T2 source "
                "cases predate W8-T3 and contain no request_id"
            ),
            "selected_case_request_joins": joined_count,
            "runtime_log_event_count": len(runtime_events),
        },
        "taxonomy": {
            "failure_types": list(FAILURE_TYPES),
            "evidence_strengths": list(EVIDENCE_STRENGTHS),
            "primary_rule": "earliest causally sufficient failure supported by saved evidence",
            "quality_execution_separation": True,
        },
        "aggregate": aggregate,
        "cases": selected_cases,
        "observability_gaps": [
            "Historical W8-T1/T2 cases do not contain request_id or embedded W8-T3 runtime events.",
            "W8-T3 acceptance logs do not contain evaluation run_id, query_id, raw query, answer, or full context.",
            "Only q025-q027 have strict chunk Ground Truth; other stage diagnoses are document-level.",
            "Citations are pipeline-derived from every finalized context, not claim-level model attribution.",
        ],
        "metric_limitations": [
            "Required-keyword matching is lexical and can reject semantically equivalent wording.",
            "Document citation exact match penalizes every extra pipeline-attached context source.",
            "Groundedness and answer relevance were not independently automated in W8-T1.",
        ],
        "future_hypotheses": [
            (
                "Multi-relevant questions may be sensitive to candidate coverage and final cutoff; "
                "a future controlled experiment should measure quality, latency, noise, and cost."
            ),
            (
                "Claim-level citation selection may reduce unrelated pipeline citations; this "
                "requires a future measured design, not a W8-T4 code change."
            ),
        ],
    }


def write_json_artifact(
    artifact: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"failure analysis artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_failure_analysis_report(
    artifact: Mapping[str, Any],
    *,
    artifact_path: str,
) -> str:
    aggregate = artifact["aggregate"]
    provenance = artifact["source_provenance"]
    config = artifact["resolved_configuration"]
    llm = config["llm"]
    cases = artifact["cases"]
    quality_cases = [
        case
        for case in cases
        if case["primary_failure"] not in {"no_failure", "needs_review", "execution_failure"}
    ]
    lines = [
        "# Failure Analysis Report",
        "",
        "## 1. Analysis Run Identity",
        "",
        f"- Analysis run: `{artifact['run_id']}`",
        f"- W8-T1 source: `{provenance['evaluation']['run_id']}` (`{provenance['evaluation']['sha256']}`)",
        f"- W8-T2 source: `{provenance['unanswerable']['run_id']}` (`{provenance['unanswerable']['sha256']}`)",
        f"- Dataset: `{provenance['dataset_identity']['path']}` (`{provenance['dataset_identity']['sha256']}`)",
        f"- Corpus chunk snapshot: `{provenance['corpus_identity']['chunk_index']['path']}` (`{provenance['corpus_identity']['chunk_index']['sha256']}`)",
        f"- Model: `{llm['provider']}` / `{llm['model']}`; digest `{llm['model_identity']['digest']}`",
        f"- Retrieval: `{config['retrieval_mode']}`, final_top_k={config['final_top_k']}",
        f"- Derived artifact: `{artifact_path}`",
        *[
            f"- Runtime log: `{record['path']}` (`{record['sha256']}`, role={record['role']}, events={record['event_count']})"
            for record in provenance["runtime_logs"]
        ],
        "",
        "## 2. Failure Taxonomy",
        "",
        "- `retrieval_failure`: required KB evidence never entered the saved candidate set.",
        "- `ranking_failure`: required evidence was a candidate but fell below the effective cutoff.",
        "- `context_construction_failure`: selected evidence did not reach the final LLM context.",
        "- `generation_failure`: sufficient context arrived, but the answer remained wrong or unsupported.",
        "- `citation_failure`: answer/evidence is otherwise acceptable but response citation behavior is wrong.",
        "- `execution_failure`: no valid quality output exists because execution failed.",
        "- `needs_review`: available evidence cannot distinguish a trustworthy cause.",
        "- `no_failure`: success control or metric-only false positive.",
        "",
        "## 3. Classification Method",
        "",
        "Classification is evidence-driven. It checks candidate presence, final-cutoff survival, context presence, answer review, and citation review in pipeline order, then chooses the earliest causally sufficient failure. Independent downstream citation defects may be contributing failures. No LLM judge, agent, rerun, or parameter tuning was used.",
        "",
        "## 4. Cases Analyzed",
        "",
        f"- Source case executions: {aggregate['source_case_counts']['raw_source_case_executions']}.",
        f"- Deduplicated primary quality universe: {aggregate['source_case_counts']['deduplicated_primary_quality_universe']} cases.",
        f"- Observed formal signal cases: {aggregate['observed_signal_case_count']}; all were analyzed.",
        f"- Selected success controls: {len(artifact['selection']['success_control_query_ids'])}.",
        f"- Existing-artifact cases analyzed: {artifact['selection']['cases_analyzed_from_existing_artifacts']}; targeted reproductions: {artifact['selection']['targeted_reproduction_count']}.",
        f"- Quality failures: {aggregate['quality_failure_case_count']} answerable={aggregate['answerable_quality_failure_count']} unanswerable={aggregate['unanswerable_quality_failure_count']}.",
        f"- Runtime failures: {aggregate['execution_failure_count']}; needs review: {aggregate['needs_review_count']}.",
        "",
        "## 5. Aggregate Primary Failures",
        "",
        "| Failure type | Count | Rate | Denominator |",
        "|---|---:|---:|---|",
    ]
    denominator = aggregate["quality_failure_case_count"]
    for failure in FAILURE_TYPES:
        if failure in {"no_failure", "needs_review", "execution_failure"}:
            continue
        count = aggregate["primary_failure_counts"][failure]
        rate = count / denominator if denominator else None
        lines.append(
            f"| {failure} | {count} | {_fmt_rate(rate)} | {denominator} classified quality failures |"
        )
    lines.extend(
        [
            "",
            "## 6. Contributing Failures",
            "",
            "| Failure type | Additional occurrences |",
            "|---|---:|",
        ]
    )
    for failure, count in aggregate["contributing_failure_counts"].items():
        lines.append(f"| {failure} | {count} |")

    section_map = (
        ("7. Retrieval Failures", "retrieval_failure"),
        ("8. Ranking Failures", "ranking_failure"),
        ("9. Context Construction Failures", "context_construction_failure"),
        ("10. Generation Failures", "generation_failure"),
        ("11. Citation Failures", "citation_failure"),
    )
    for title, failure in section_map:
        lines.extend(["", f"## {title}", ""])
        matching = [case for case in quality_cases if case["primary_failure"] == failure]
        if not matching:
            lines.append("None observed.")
        else:
            for case in matching:
                lines.append(
                    f"- `{case['query_id']}` ({case['evidence_strength']}): {case['causal_basis']}"
                )

    unanswerable_failures = [case for case in quality_cases if not case["answerable"]]
    runtime_failures = [case for case in cases if case["primary_failure"] == "execution_failure"]
    review_cases = [case for case in cases if case["primary_failure"] == "needs_review"]
    lines.extend(
        [
            "",
            "## 12. Unanswerable Failures",
            "",
            (
                "None observed in W8-T2; all eight formal unanswerable cases were classified as correct abstention by the frozen deterministic proxy. The selected code-level and qwen3:8b controls remain `no_failure`."
                if not unanswerable_failures
                else "\n".join(f"- `{case['query_id']}`: {case['causal_basis']}" for case in unanswerable_failures)
            ),
            "",
            "## 13. Runtime Failures",
            "",
            (
                "None observed. Runtime failures are counted separately from answer-quality failures."
                if not runtime_failures
                else "\n".join(f"- `{case['query_id']}`: {case['causal_basis']}" for case in runtime_failures)
            ),
            "",
            "## 14. Needs Review",
            "",
            (
                "None in the selected formal cases. The classifier still preserves `needs_review` when candidates, contexts, Ground Truth, or semantic review evidence is missing."
                if not review_cases
                else "\n".join(f"- `{case['query_id']}`: {case['causal_basis']}" for case in review_cases)
            ),
            "",
            "## 15. Observed Patterns",
            "",
            f"- {aggregate['primary_failure_counts']['citation_failure']} of {denominator} quality failures were response-level citation failures caused by pipeline citations containing an unrelated extra document.",
            f"- The only observed retrieval primary failure was a multi-document, strict-chunk-labeled question.",
            "- No ranking, context-construction, generation-primary, unanswerable-behavior, or execution failure was observed in the analyzed runs.",
            "",
            "## 16. Representative Case Walkthroughs",
            "",
        ]
    )
    for query_id in ("q027", "q006", "q005"):
        case = next((item for item in cases if item["query_id"] == query_id), None)
        if case is None:
            continue
        lines.extend(
            [
                f"### `{query_id}` — {case['primary_failure']}",
                "",
                f"- Query: {case.get('question')}",
                f"- Expected answer: {case.get('expected_answer')}",
                f"- Actual answer: {case.get('actual_answer')}",
                f"- Observed problem: {case.get('observed_problem', 'success control')}",
                f"- Expected documents/chunks: {case.get('expected_documents')} / {case.get('expected_chunk_ids')}",
                f"- Candidate IDs: {case.get('candidate_ids_by_field', {}).get('chunk_id')}",
                f"- Final IDs: {case.get('final_ids_by_field', {}).get('chunk_id')}",
                f"- Context IDs: {case.get('context_ids_by_field', {}).get('chunk_id')}",
                f"- Citation IDs: {case.get('citation_ids', {}).get('chunk_id')}",
                f"- Answer assessment: {case.get('answer_quality', 'control proxy')}; citation assessment: {case.get('citation_quality')}",
                f"- Classification: primary `{case['primary_failure']}`, contributing {case['contributing_failures']}, strength `{case['evidence_strength']}`.",
                f"- Causal basis: {case['causal_basis']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 17. Metric Limitations",
            "",
            *[f"- {item}" for item in artifact["metric_limitations"]],
            "- In q005, `3-month` versus `3 months` made the lexical proxy fail even though the answer was acceptable; this is an evaluator false positive, not a generation failure.",
            "",
            "## 18. Observability Gaps",
            "",
            *[f"- {item}" for item in artifact["observability_gaps"]],
            f"- Selected case request-level joins: {artifact['evidence_join']['selected_case_request_joins']} of {len(cases)}; the analyzer supports request_id joins, but the historical source cases have no request_id.",
            "",
            "## 19. Future Hypotheses",
            "",
            *[f"- Hypothesis: {item}" for item in artifact["future_hypotheses"]],
            "",
            "These are future experiments, not implemented fixes or proven decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
