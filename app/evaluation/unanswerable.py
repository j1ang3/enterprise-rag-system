import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.generation import normalize_lexical_text
from app.evaluation.rag import RAGEvaluationRunner


UNANSWERABLE_ARTIFACT_VERSION = 1
UNANSWERABLE_CASE_SCHEMA_VERSION = "unanswerable_cases.v1"
ALLOWED_CASE_FIELDS = {
    "query_id",
    "question",
    "answerable",
    "source",
    "case_type",
    "expected_behavior",
    "absence_basis",
}
ALLOWED_ABSENCE_FIELDS = {
    "full_source_review",
    "full_chunk_snapshot_review",
    "lexical_terms",
    "expected_absent_phrases",
    "notes",
}
ALLOWED_CASE_TYPES = {
    "absent_entity_fact",
    "missing_numeric_fact",
    "external_knowledge",
    "insufficient_related_evidence",
    "unclassified",
}
ALLOWED_SOURCES = {"stable_dataset", "w8_t2_supplement"}
QUERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
OUTCOMES = (
    "correct_abstention",
    "contaminated_abstention",
    "unsupported_answer",
    "needs_review",
    "execution_failure",
)


@dataclass(frozen=True)
class AbsenceBasis:
    full_source_review: bool
    full_chunk_snapshot_review: bool
    lexical_terms: Tuple[str, ...]
    expected_absent_phrases: Tuple[str, ...]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_source_review": self.full_source_review,
            "full_chunk_snapshot_review": self.full_chunk_snapshot_review,
            "lexical_terms": list(self.lexical_terms),
            "expected_absent_phrases": list(self.expected_absent_phrases),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class UnanswerableCase:
    query_id: str
    question: str
    source: str
    case_type: str
    expected_behavior: str
    absence_basis: AbsenceBasis

    def to_evaluation_case(self) -> EvaluationCase:
        return EvaluationCase(
            query_id=self.query_id,
            question=self.question,
            expected_answer=(
                "The current knowledge base does not contain enough information "
                "to answer this question reliably."
            ),
            expected_sources=(),
            expected_source_match="any",
            expected_citation_chunk_ids=(),
            expected_keywords=(),
            category="no_answer",
            difficulty="controlled",
            answerable=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "answerable": False,
            "source": self.source,
            "case_type": self.case_type,
            "expected_behavior": self.expected_behavior,
            "absence_basis": self.absence_basis.to_dict(),
        }


def _non_empty_string(value: object, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} {field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, field: str, location: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} {field} must be a non-empty list of strings")
    values = tuple(
        _non_empty_string(item, field=field, location=location) for item in value
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{location} {field} must not contain duplicates")
    return values


def _normalize_unanswerable_case(
    raw: object,
    *,
    position: int,
    line_number: int,
) -> UnanswerableCase:
    location = f"unanswerable case {position} (line {line_number})"
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be a JSON object")
    unknown = sorted(set(raw) - ALLOWED_CASE_FIELDS)
    if unknown:
        raise ValueError(f"{location} contains unknown fields: {', '.join(unknown)}")
    if raw.get("answerable") is not False:
        raise ValueError(f"{location} answerable must be explicitly false")

    source = _non_empty_string(raw.get("source"), field="source", location=location)
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"{location} source is unsupported")
    case_type = _non_empty_string(
        raw.get("case_type"), field="case_type", location=location
    )
    if case_type not in ALLOWED_CASE_TYPES:
        raise ValueError(f"{location} case_type is unsupported")
    if raw.get("expected_behavior") != "abstain":
        raise ValueError(f"{location} expected_behavior must be 'abstain'")

    absence = raw.get("absence_basis")
    if not isinstance(absence, dict):
        raise ValueError(f"{location} absence_basis must be an object")
    unknown_absence = sorted(set(absence) - ALLOWED_ABSENCE_FIELDS)
    if unknown_absence:
        raise ValueError(
            f"{location} absence_basis contains unknown fields: "
            f"{', '.join(unknown_absence)}"
        )
    if absence.get("full_source_review") is not True:
        raise ValueError(f"{location} requires full_source_review=true")
    if absence.get("full_chunk_snapshot_review") is not True:
        raise ValueError(f"{location} requires full_chunk_snapshot_review=true")

    query_id = _non_empty_string(
        raw.get("query_id"), field="query_id", location=location
    )
    if not QUERY_ID_PATTERN.fullmatch(query_id):
        raise ValueError(f"{location} query_id contains unsupported characters")

    return UnanswerableCase(
        query_id=query_id,
        question=_non_empty_string(
            raw.get("question"), field="question", location=location
        ),
        source=source,
        case_type=case_type,
        expected_behavior="abstain",
        absence_basis=AbsenceBasis(
            full_source_review=True,
            full_chunk_snapshot_review=True,
            lexical_terms=_string_tuple(
                absence.get("lexical_terms"),
                field="absence_basis.lexical_terms",
                location=location,
            ),
            expected_absent_phrases=_string_tuple(
                absence.get("expected_absent_phrases"),
                field="absence_basis.expected_absent_phrases",
                location=location,
            ),
            notes=_non_empty_string(
                absence.get("notes"),
                field="absence_basis.notes",
                location=location,
            ),
        ),
    )


def load_unanswerable_cases(
    path: Path,
    *,
    stable_dataset: EvaluationDataset,
) -> Tuple[UnanswerableCase, ...]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"unanswerable case file not found: {resolved}")
    cases = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid unanswerable JSONL at {resolved}:{line_number}"
            ) from exc
        cases.append(
            _normalize_unanswerable_case(
                raw,
                position=len(cases) + 1,
                line_number=line_number,
            )
        )
    if not cases:
        raise ValueError("unanswerable case file must not be empty")

    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("unanswerable case file contains duplicate query_id values")

    stable_by_id = {case.query_id: case for case in stable_dataset.cases}
    for case in cases:
        stable = stable_by_id.get(case.query_id)
        if case.source == "stable_dataset":
            if stable is None:
                raise ValueError(
                    f"stable unanswerable case is missing from dataset: {case.query_id}"
                )
            if stable.answerable:
                raise ValueError(
                    f"stable unanswerable case is labeled answerable: {case.query_id}"
                )
            if stable.question != case.question:
                raise ValueError(
                    f"stable unanswerable question drifted: {case.query_id}"
                )
        elif stable is not None:
            raise ValueError(
                f"supplemental query_id collides with stable dataset: {case.query_id}"
            )
    return tuple(cases)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_recorded_path(project_root: Path, recorded: object, *, field: str) -> Path:
    value = _non_empty_string(recorded, field=field, location="manifest")
    path = (project_root / value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"manifest {field} does not exist: {path}")
    return path


def _validate_identity_file(
    project_root: Path,
    record: object,
    *,
    field: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest {field} must be an object")
    path = _resolve_recorded_path(project_root, record.get("path"), field=field)
    expected_hash = _non_empty_string(
        record.get("sha256"), field=f"{field}.sha256", location="manifest"
    )
    if file_sha256(path) != expected_hash:
        raise ValueError(f"manifest identity mismatch: {field}")
    return path


def load_unanswerable_manifest(
    path: Path,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"unanswerable manifest not found: {resolved}")
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("unanswerable manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("unanswerable manifest must be an object")
    if manifest.get("manifest_version") != 1 or manifest.get("task") != "W8-T2":
        raise ValueError("unanswerable manifest identity is invalid")
    if manifest.get("created_before_formal_results") is not True:
        raise ValueError("unanswerable rubric must be frozen before formal results")

    stable_path = _validate_identity_file(
        project_root, manifest.get("stable_dataset"), field="stable_dataset"
    )
    case_path = _validate_identity_file(
        project_root, manifest.get("case_file"), field="case_file"
    )
    w8_t1_path = _validate_identity_file(
        project_root,
        manifest.get("source_w8_t1_artifact"),
        field="source_w8_t1_artifact",
    )
    w7_path = _validate_identity_file(
        project_root,
        manifest.get("frozen_w7_manifest"),
        field="frozen_w7_manifest",
    )

    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("unanswerable manifest corpus must be an object")
    chunk_path = _validate_identity_file(
        project_root, corpus.get("chunk_index"), field="corpus.chunk_index"
    )
    documents = corpus.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("unanswerable manifest requires corpus documents")
    document_paths = [
        _validate_identity_file(
            project_root,
            record,
            field=f"corpus.documents[{position}]",
        )
        for position, record in enumerate(documents, start=1)
    ]

    rubric = manifest.get("rubric")
    if not isinstance(rubric, Mapping):
        raise ValueError("unanswerable manifest rubric must be an object")
    if rubric.get("version") != "w8-t2-abstention-rubric.v1":
        raise ValueError("unsupported unanswerable rubric version")
    if rubric.get("classification_method") != "deterministic_rule_based_proxy":
        raise ValueError("unsupported unanswerable classification method")
    if rubric.get("llm_judge") is not False:
        raise ValueError("W8-T2 manifest must not enable an LLM judge")
    for field in (
        "abstention_phrases",
        "contamination_markers",
        "contrast_markers",
        "direct_claim_verbs",
    ):
        _string_tuple(rubric.get(field), field=f"rubric.{field}", location="manifest")

    return {
        "manifest": manifest,
        "manifest_path": resolved,
        "manifest_sha256": file_sha256(resolved),
        "stable_dataset_path": stable_path,
        "case_file_path": case_path,
        "source_w8_t1_artifact_path": w8_t1_path,
        "w7_manifest_path": w7_path,
        "chunk_index_path": chunk_path,
        "document_paths": tuple(document_paths),
    }


def validate_w8_t1_source_artifact(
    path: Path,
    *,
    expected_dataset_sha256: str,
) -> Dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("source W8-T1 artifact is invalid JSON") from exc
    if not isinstance(artifact, dict):
        raise ValueError("source W8-T1 artifact must be an object")
    if artifact.get("task") != "W8-T1" or artifact.get("formal") is not True:
        raise ValueError("source artifact is not a formal W8-T1 run")
    dataset = artifact.get("dataset_identity")
    if not isinstance(dataset, Mapping) or dataset.get("sha256") != expected_dataset_sha256:
        raise ValueError("source W8-T1 dataset identity mismatch")
    config = artifact.get("resolved_configuration")
    llm = config.get("llm") if isinstance(config, Mapping) else None
    if not isinstance(llm, Mapping):
        raise ValueError("source W8-T1 LLM identity is missing")
    if llm.get("provider") != "ollama" or llm.get("model") != "qwen3:8b":
        raise ValueError("source W8-T1 model is not Ollama qwen3:8b")
    identity = llm.get("model_identity")
    if not isinstance(identity, Mapping) or not identity.get("digest"):
        raise ValueError("source W8-T1 resolved model identity is missing")
    return artifact


def verify_absence_against_corpus(
    cases: Sequence[UnanswerableCase],
    *,
    document_paths: Sequence[Path],
    chunk_index_path: Path,
) -> Dict[str, Dict[str, Any]]:
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in document_paths
    )
    try:
        chunks = json.loads(chunk_index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("chunk snapshot is invalid JSON") from exc
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunk snapshot must be a non-empty list")
    chunk_text = "\n".join(
        str(chunk.get("content", "")) for chunk in chunks if isinstance(chunk, Mapping)
    )
    normalized_source = normalize_lexical_text(source_text)
    normalized_chunks = normalize_lexical_text(chunk_text)

    verified: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        phrase_checks = []
        for phrase in case.absence_basis.expected_absent_phrases:
            normalized = normalize_lexical_text(phrase)
            source_count = normalized_source.count(normalized)
            chunk_count = normalized_chunks.count(normalized)
            phrase_checks.append(
                {
                    "phrase": phrase,
                    "source_occurrences": source_count,
                    "chunk_occurrences": chunk_count,
                    "absent": source_count == 0 and chunk_count == 0,
                }
            )
        if not all(check["absent"] for check in phrase_checks):
            raise ValueError(
                f"unanswerable absence phrase is now present: {case.query_id}"
            )
        term_checks = []
        for term in case.absence_basis.lexical_terms:
            normalized = normalize_lexical_text(term)
            term_checks.append(
                {
                    "term": term,
                    "source_occurrences": normalized_source.count(normalized),
                    "chunk_occurrences": normalized_chunks.count(normalized),
                }
            )
        verified[case.query_id] = {
            "status": "verified_absent_for_current_corpus_snapshot",
            "case_type": case.case_type,
            "source": case.source,
            "methods": [
                "full_source_document_review",
                "full_chunk_snapshot_review",
                "lexical_phrase_check",
                "frozen_semantic_retrieval_evidence_saved_at_run_time",
            ],
            "phrase_checks": phrase_checks,
            "term_checks": term_checks,
            "notes": case.absence_basis.notes,
            "limitation": (
                "Absence is established for the recorded finite corpus snapshot; "
                "lexical or semantic search alone is not a logical proof of absence."
            ),
        }
    return verified


def _rubric_string_list(rubric: Mapping[str, Any], field: str) -> Tuple[str, ...]:
    value = rubric.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"rubric {field} must be a list of strings")
    return tuple(item.casefold() for item in value)


def detect_abstention_signals(
    answer: str,
    rubric: Mapping[str, Any],
) -> list[str]:
    normalized = normalize_lexical_text(answer)
    return [
        phrase
        for phrase in _rubric_string_list(rubric, "abstention_phrases")
        if phrase in normalized
    ]


def _specific_claim_evidence(
    answer: str,
    rubric: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = normalize_lexical_text(answer)
    words = re.findall(r"[a-z0-9%$]+", normalized)
    number_matches = re.findall(r"(?:\$\s*)?\d+(?:\.\d+)?\s*%?", normalized)
    direct_verbs = [
        verb
        for verb in _rubric_string_list(rubric, "direct_claim_verbs")
        if re.search(rf"\b{re.escape(verb)}\b", normalized)
    ]
    minimum_words = rubric.get("minimum_direct_claim_words")
    if isinstance(minimum_words, bool) or not isinstance(minimum_words, int):
        raise ValueError("rubric minimum_direct_claim_words must be an integer")
    return {
        "word_count": len(words),
        "number_matches": number_matches,
        "direct_claim_verbs": direct_verbs,
        "specific_claim_detected": bool(number_matches)
        or (len(words) >= minimum_words and bool(direct_verbs)),
    }


def classify_unanswerable_result(
    result: Mapping[str, Any],
    *,
    rubric: Mapping[str, Any],
) -> Dict[str, Any]:
    if result.get("status") != "success":
        return {
            "outcome": "execution_failure",
            "method": "deterministic_rule_based_proxy",
            "outcome_evidence": {
                "error": result.get("error"),
                "reason": "No valid behavioral output was available.",
            },
            "citation_evaluation": None,
            "review_status": "not_applicable_execution_failure",
            "review_label": None,
            "review_notes": None,
        }
    actual = result.get("actual")
    if not isinstance(actual, Mapping):
        raise ValueError("successful result must contain actual output")
    answer = actual.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("successful result answer must be non-empty")

    signals = detect_abstention_signals(answer, rubric)
    normalized = normalize_lexical_text(answer)
    claim = _specific_claim_evidence(answer, rubric)
    speculation = [
        marker
        for marker in _rubric_string_list(rubric, "contamination_markers")
        if marker in normalized
    ]
    contrast = [
        marker
        for marker in _rubric_string_list(rubric, "contrast_markers")
        if re.search(rf"\b{re.escape(marker)}\b", normalized)
    ]

    if signals and claim["specific_claim_detected"] and (speculation or contrast):
        outcome = "contaminated_abstention"
        reason = "An abstention signal is followed by specific speculative or contrasting factual content."
    elif signals:
        outcome = "correct_abstention"
        reason = "The response explicitly signals insufficient knowledge-base evidence without a detected contaminating claim."
    elif claim["specific_claim_detected"]:
        outcome = "unsupported_answer"
        reason = "The response contains a specific factual-answer signal without an abstention signal."
    else:
        outcome = "needs_review"
        reason = "The deterministic rubric found neither a clear abstention nor a sufficiently specific factual-answer signal."

    citations = actual.get("citations")
    if not isinstance(citations, list):
        raise ValueError("successful result citations must be a list")
    cited_ids = [
        citation.get("chunk_id")
        for citation in citations
        if isinstance(citation, Mapping) and isinstance(citation.get("chunk_id"), str)
    ]
    model_mentioned_ids = [chunk_id for chunk_id in cited_ids if chunk_id in answer]
    misleading = outcome in {
        "unsupported_answer",
        "contaminated_abstention",
    } and bool(cited_ids)
    if not cited_ids:
        citation_outcome = "no_pipeline_citations"
    elif misleading:
        citation_outcome = "misleading_pipeline_citation_proxy"
    elif outcome == "correct_abstention":
        citation_outcome = "pipeline_sources_on_abstention"
    else:
        citation_outcome = "needs_review"

    return {
        "outcome": outcome,
        "method": "deterministic_rule_based_proxy",
        "outcome_evidence": {
            "reason": reason,
            "answer_mode": actual.get("answer_mode"),
            "actual_model": actual.get("model"),
            "abstention_signals": signals,
            "specific_claim": claim,
            "speculation_markers": speculation,
            "contrast_markers": contrast,
        },
        "citation_evaluation": {
            "origin": "pipeline_derived_from_final_contexts",
            "citation_count": len(cited_ids),
            "cited_chunk_ids": cited_ids,
            "model_mentioned_chunk_ids": model_mentioned_ids,
            "outcome": citation_outcome,
            "misleading_citation_proxy": misleading,
        },
        "review_status": "not_manually_reviewed",
        "review_label": None,
        "review_notes": (
            "Rule-based proxy only; raw answer and retrieval evidence remain available for review."
        ),
    }


def classify_answerable_control(
    result: Mapping[str, Any],
    *,
    rubric: Mapping[str, Any],
) -> Dict[str, Any]:
    if result.get("status") != "success":
        return {
            "status": "execution_failure",
            "false_abstention": None,
            "abstention_signals": [],
        }
    actual = result.get("actual")
    if not isinstance(actual, Mapping) or not isinstance(actual.get("answer"), str):
        raise ValueError("successful control must contain an answer")
    signals = detect_abstention_signals(actual["answer"], rubric)
    false_abstention = actual.get("answer_mode") == "no_context" or bool(signals)
    return {
        "status": "evaluated",
        "false_abstention": false_abstention,
        "abstention_signals": signals,
        "answer_mode": actual.get("answer_mode"),
        "actual_model": actual.get("model"),
    }


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def aggregate_unanswerable_results(
    unanswerable_results: Sequence[Mapping[str, Any]],
    control_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    successful = [row for row in unanswerable_results if row.get("status") == "success"]
    failed = [row for row in unanswerable_results if row.get("status") != "success"]
    outcome_counts = {outcome: 0 for outcome in OUTCOMES}
    for row in unanswerable_results:
        evaluation = row.get("behavior_evaluation")
        outcome = evaluation.get("outcome") if isinstance(evaluation, Mapping) else None
        if outcome not in outcome_counts:
            raise ValueError(f"unknown unanswerable outcome: {outcome}")
        outcome_counts[str(outcome)] += 1

    successful_count = len(successful)
    unsupported_count = (
        outcome_counts["unsupported_answer"]
        + outcome_counts["contaminated_abstention"]
    )
    misleading_count = sum(
        bool(row["behavior_evaluation"]["citation_evaluation"]["misleading_citation_proxy"])
        for row in successful
        if row["behavior_evaluation"]["citation_evaluation"] is not None
    )

    llm_rows = [
        row
        for row in successful
        if isinstance(row.get("actual"), Mapping)
        and row["actual"].get("answer_mode") == "llm"
    ]
    llm_outcomes = {outcome: 0 for outcome in OUTCOMES if outcome != "execution_failure"}
    for row in llm_rows:
        llm_outcomes[row["behavior_evaluation"]["outcome"]] += 1

    successful_controls = [
        row
        for row in control_results
        if row.get("status") == "success"
        and row.get("control_evaluation", {}).get("status") == "evaluated"
    ]
    failed_controls = [row for row in control_results if row.get("status") != "success"]
    false_abstentions = sum(
        bool(row["control_evaluation"]["false_abstention"])
        for row in successful_controls
    )

    answer_mode_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}
    for row in [*unanswerable_results, *control_results]:
        actual = row.get("actual")
        if not isinstance(actual, Mapping):
            continue
        mode = actual.get("answer_mode")
        if mode:
            answer_mode_counts[str(mode)] = answer_mode_counts.get(str(mode), 0) + 1
        model = actual.get("model")
        if model:
            model_counts[str(model)] = model_counts.get(str(model), 0) + 1

    return {
        "unanswerable": {
            "total": len(unanswerable_results),
            "successful": successful_count,
            "failed": len(failed),
            "outcome_counts": outcome_counts,
            "strict_abstention_rate": _rate(
                outcome_counts["correct_abstention"], successful_count
            ),
            "contaminated_abstention_rate": _rate(
                outcome_counts["contaminated_abstention"], successful_count
            ),
            "unsupported_answer_count_including_contaminated": unsupported_count,
            "unsupported_answer_rate_including_contaminated": _rate(
                unsupported_count, successful_count
            ),
            "misleading_citation_count": misleading_count,
            "misleading_citation_rate": _rate(misleading_count, successful_count),
            "ambiguous_review_rate": _rate(
                outcome_counts["needs_review"], successful_count
            ),
            "execution_failure_rate": _rate(len(failed), len(unanswerable_results)),
            "failed_query_ids": [row["query_id"] for row in failed],
        },
        "llm_unanswerable_subset": {
            "sample_count": len(llm_rows),
            "outcome_counts": llm_outcomes,
            "strict_abstention_rate": _rate(
                llm_outcomes["correct_abstention"], len(llm_rows)
            ),
            "unsupported_answer_rate_including_contaminated": _rate(
                llm_outcomes["unsupported_answer"]
                + llm_outcomes["contaminated_abstention"],
                len(llm_rows),
            ),
        },
        "answerable_controls": {
            "total": len(control_results),
            "successful": len(successful_controls),
            "failed": len(failed_controls),
            "false_abstention_count": false_abstentions,
            "false_abstention_rate": _rate(
                false_abstentions, len(successful_controls)
            ),
            "failed_query_ids": [row["query_id"] for row in failed_controls],
        },
        "execution": {
            "answer_mode_counts": answer_mode_counts,
            "model_counts": model_counts,
        },
        "denominators": {
            "behavior": "successful unanswerable system outputs",
            "llm_behavior": "successful unanswerable outputs with answer_mode=llm",
            "execution_failure": "all unanswerable cases",
            "control_false_abstention": "successful answerable control outputs",
        },
    }


class UnanswerableEvaluationRunner:
    def __init__(
        self,
        base_runner: RAGEvaluationRunner,
        *,
        rubric: Mapping[str, Any],
    ) -> None:
        self.base_runner = base_runner
        self.rubric = dict(rubric)

    def _run_unanswerable_case(
        self,
        case: UnanswerableCase,
        absence_verification: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = self.base_runner.run_case(case.to_evaluation_case())
        result["case_source"] = case.source
        result["case_type"] = case.case_type
        result["absence_verification"] = dict(absence_verification)
        try:
            result["behavior_evaluation"] = classify_unanswerable_result(
                result,
                rubric=self.rubric,
            )
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = {
                "failure_stage": "classification",
                "category": "classifier_error",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            result["behavior_evaluation"] = classify_unanswerable_result(
                result,
                rubric=self.rubric,
            )
        return result

    def _run_control(self, case: EvaluationCase) -> Dict[str, Any]:
        result = self.base_runner.run_case(case)
        try:
            result["control_evaluation"] = classify_answerable_control(
                result,
                rubric=self.rubric,
            )
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = {
                "failure_stage": "classification",
                "category": "classifier_error",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            result["control_evaluation"] = {
                "status": "execution_failure",
                "false_abstention": None,
                "abstention_signals": [],
            }
        return result

    def run(
        self,
        *,
        unanswerable_cases: Sequence[UnanswerableCase],
        absence_verification: Mapping[str, Mapping[str, Any]],
        control_cases: Sequence[EvaluationCase],
        run_id: str,
        run_metadata: Mapping[str, Any],
        source_identities: Mapping[str, Any],
        project_root: Path,
    ) -> Dict[str, Any]:
        unanswerable_results = [
            self._run_unanswerable_case(
                case,
                absence_verification=absence_verification[case.query_id],
            )
            for case in unanswerable_cases
        ]
        control_results = [self._run_control(case) for case in control_cases]
        aggregate = aggregate_unanswerable_results(
            unanswerable_results,
            control_results,
        )
        return {
            "artifact_version": UNANSWERABLE_ARTIFACT_VERSION,
            "task": "W8-T2",
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "formal": self.base_runner.configuration.formal,
            "status": (
                "completed"
                if not aggregate["unanswerable"]["failed"]
                and not aggregate["answerable_controls"]["failed"]
                else "completed_with_failures"
            ),
            "run_metadata": dict(run_metadata),
            "source_identities": dict(source_identities),
            "resolved_configuration": self.base_runner.configuration.resolved_dict(
                project_root=project_root
            ),
            "rubric": dict(self.rubric),
            "aggregate": aggregate,
            "unanswerable_results": unanswerable_results,
            "answerable_control_results": control_results,
        }


def render_unanswerable_report(
    artifact: Mapping[str, Any],
    *,
    artifact_path: str,
) -> str:
    aggregate = artifact["aggregate"]
    unanswerable = aggregate["unanswerable"]
    llm_subset = aggregate["llm_unanswerable_subset"]
    controls = aggregate["answerable_controls"]
    config = artifact["resolved_configuration"]
    llm = config["llm"]
    rubric = artifact["rubric"]
    identities = artifact["source_identities"]
    dataset = identities["stable_dataset"]
    case_file = identities["unanswerable_case_file"]
    corpus = identities["corpus"]
    lines = [
        "# Unanswerable Evaluation Report",
        "",
        f"- Run ID: `{artifact['run_id']}`",
        f"- Formal: `{str(artifact['formal']).lower()}`",
        f"- Provider / model: `{llm['provider']}` / `{llm['model']}`",
        f"- Model digest: `{llm.get('model_identity', {}).get('digest')}`",
        f"- Rubric: `{rubric['version']}` ({rubric['classification_method']})",
        f"- Retrieval: `{config['retrieval_mode']}`; final top_k={config['final_top_k']}",
        f"- Stable dataset: `{dataset['path']}` (`{dataset['sha256']}`, {dataset['query_count']} cases)",
        f"- Unanswerable case set: `{case_file['path']}` (`{case_file['sha256']}`, {len(case_file['query_ids'])} cases)",
        f"- Corpus: {len(corpus['documents'])} documents / {corpus['indexed_chunk_count']} chunks; chunk index `{corpus['chunk_index']['path']}` (`{corpus['chunk_index']['sha256']}`)",
        f"- Artifact: `{artifact_path}`",
        "",
        "## What Counts as Unanswerable",
        "",
        "A case is KB-unanswerable only when the requested fact is absent from the recorded corpus snapshot. A retrieval miss alone is not absence evidence. Full source/chunk review and lexical phrase checks were frozen before this run; runtime retrieval evidence is preserved for inspection.",
        "",
        "## Rubric",
        "",
        "- Correct abstention: explicit knowledge/context insufficiency and no detected unsupported fact.",
        "- Contaminated abstention: insufficiency statement followed by a specific guess or unsupported claim.",
        "- Unsupported answer: a specific answer without a knowledge-base insufficiency statement.",
        "- Needs review: deterministic evidence is insufficient.",
        "- Execution failure: no valid behavioral output; excluded from hallucination denominators.",
        "- Classification is a rule-based proxy, not independent human Ground Truth. No LLM judge was used.",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Count / denominator | Rate |",
        "|---|---:|---:|",
        f"| Strict system abstention | {unanswerable['outcome_counts']['correct_abstention']} / {unanswerable['successful']} | {_fmt_rate(unanswerable['strict_abstention_rate'])} |",
        f"| Contaminated abstention | {unanswerable['outcome_counts']['contaminated_abstention']} / {unanswerable['successful']} | {_fmt_rate(unanswerable['contaminated_abstention_rate'])} |",
        f"| Unsupported including contaminated | {unanswerable['unsupported_answer_count_including_contaminated']} / {unanswerable['successful']} | {_fmt_rate(unanswerable['unsupported_answer_rate_including_contaminated'])} |",
        f"| Misleading citation proxy | {unanswerable['misleading_citation_count']} / {unanswerable['successful']} | {_fmt_rate(unanswerable['misleading_citation_rate'])} |",
        f"| Needs review | {unanswerable['outcome_counts']['needs_review']} / {unanswerable['successful']} | {_fmt_rate(unanswerable['ambiguous_review_rate'])} |",
        f"| Execution failure | {unanswerable['failed']} / {unanswerable['total']} | {_fmt_rate(unanswerable['execution_failure_rate'])} |",
        f"| Answerable-control false abstention | {controls['false_abstention_count']} / {controls['successful']} | {_fmt_rate(controls['false_abstention_rate'])} |",
        "",
        "## qwen3:8b Unanswerable Subset",
        "",
        f"- Generation-bearing unanswerable cases: {llm_subset['sample_count']}.",
        f"- Strict abstention rate: {_fmt_rate(llm_subset['strict_abstention_rate'])}.",
        f"- Unsupported-answer rate including contaminated abstention: {_fmt_rate(llm_subset['unsupported_answer_rate_including_contaminated'])}.",
        "- Code-level `no_context` cases are reported in the system aggregate but are not attributed to qwen3:8b.",
        "",
        "## Per-case Outcomes",
        "",
        "| Query | Source | Type | Answer mode | Outcome | Citations | Review |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in artifact["unanswerable_results"]:
        actual = row.get("actual") or {}
        behavior = row["behavior_evaluation"]
        citation = behavior.get("citation_evaluation") or {}
        lines.append(
            f"| {row['query_id']} | {row['case_source']} | {row['case_type']} | "
            f"{actual.get('answer_mode', 'n/a')} | {behavior['outcome']} | "
            f"{citation.get('citation_count', 0)} | {behavior['review_status']} |"
        )

    lines.extend(
        [
            "",
            "## Answerable Controls",
            "",
            "| Query | Category | Answer mode | Model | False abstention |",
            "|---|---|---|---|---|",
        ]
    )
    for row in artifact["answerable_control_results"]:
        actual = row.get("actual") or {}
        control = row["control_evaluation"]
        lines.append(
            f"| {row['query_id']} | {row['category']} | {actual.get('answer_mode', 'n/a')} | "
            f"{actual.get('model') or 'n/a'} | {control.get('false_abstention')} |"
        )

    lines.extend(["", "## Outcome Groups", ""])
    for outcome in OUTCOMES:
        query_ids = [
            row["query_id"]
            for row in artifact["unanswerable_results"]
            if row["behavior_evaluation"]["outcome"] == outcome
        ]
        label = outcome.replace("_", " ").title()
        lines.append(
            f"- {label}: "
            + (", ".join(f"`{query_id}`" for query_id in query_ids) if query_ids else "None observed.")
        )

    lines.extend(
        [
            "",
            "## Citation and External-knowledge Boundaries",
            "",
            "Production citations are attached from final contexts; citation presence does not prove model attribution. A citation is marked misleading only as a response-level proxy when an unsupported/contaminated answer is paired with pipeline sources. External facts are unsupported whenever the recorded enterprise corpus does not contain them, even if they are true in the real world.",
            "",
            "## Limitations",
            "",
            "- The case set is small and synthetic; results do not prove that the system or qwen3:8b never hallucinates.",
            "- Four stable cases are intercepted by an existing exact-evidence guard before the LLM; their success is application-level abstention, not model evidence.",
            "- Four supplemental cases are separately frozen and corpus-reviewed, but do not cover every unanswerable mechanism.",
            "- Rule-based classification can miss paraphrases or subtle contaminated claims; `needs_review` is preserved instead of forcing a label.",
            "- No LLM judge, prompt tuning, retrieval tuning, similarity threshold, structured logging, full root-cause analysis, or security attack was added.",
            "- Reproducibility records conditions and identities; Ollama text is not promised bit-for-bit identical.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
