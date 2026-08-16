import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval import classify_rank_movement


DEFAULT_SOURCE = (
    PROJECT_ROOT / "evals" / "results" / "W6-T4-retrieval-evaluation.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "results" / "W6-T5-failure-analysis.json"
EXPECTED_METHODS = ("vector", "bm25", "hybrid_rrf")
ANALYSIS_VERSION = 1


class ArtifactValidationError(ValueError):
    """The source evaluation artifact cannot support reproducible analysis."""


def load_artifact(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"source artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"source artifact is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError("source artifact root must be an object")
    return payload


def _require_mapping(
    container: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"missing or invalid {path}.{key}")
    return value


def _require_non_empty_string(
    container: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"missing or invalid {path}.{key}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_file(project_root: Path, recorded_path: str) -> Path:
    relative_path = Path(recorded_path.replace("\\", "/"))
    if relative_path.is_absolute():
        raise ArtifactValidationError("recorded evidence paths must be project-relative")
    resolved = (project_root / relative_path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ArtifactValidationError(
            f"recorded evidence path escapes project root: {recorded_path}"
        ) from exc
    return resolved


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> List[str]:
    """Validate W6-T4 identity, configuration, per-query shape, and optional files."""
    if artifact.get("task") != "W6-T4":
        raise ArtifactValidationError("source artifact task must be W6-T4")
    _require_non_empty_string(artifact, "generated_at", path="artifact")

    ground_truth = _require_mapping(artifact, "ground_truth", path="artifact")
    dataset_source = _require_non_empty_string(
        ground_truth, "source", path="artifact.ground_truth"
    )
    dataset_hash = _require_non_empty_string(
        ground_truth, "dataset_sha256", path="artifact.ground_truth"
    )
    _require_non_empty_string(
        ground_truth, "label_type", path="artifact.ground_truth"
    )

    configuration = _require_mapping(artifact, "configuration", path="artifact")
    methods = configuration.get("methods")
    if not isinstance(methods, list) or tuple(methods) != EXPECTED_METHODS:
        raise ArtifactValidationError(
            f"configuration.methods must equal {list(EXPECTED_METHODS)}"
        )
    k_values = configuration.get("k_values")
    if not isinstance(k_values, list) or not k_values or any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in k_values
    ):
        raise ArtifactValidationError("configuration.k_values must be positive integers")
    query_count = configuration.get("query_count")
    if not isinstance(query_count, int) or isinstance(query_count, bool) or query_count <= 0:
        raise ArtifactValidationError("configuration.query_count must be positive")
    for field in ("hybrid_candidate_depth", "rrf_k", "indexed_chunk_count"):
        value = configuration.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArtifactValidationError(f"configuration.{field} must be positive")

    corpus_hashes = configuration.get("bootstrap_document_sha256")
    if not isinstance(corpus_hashes, Mapping) or not corpus_hashes:
        raise ArtifactValidationError(
            "missing configuration.bootstrap_document_sha256"
        )
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or not digest
        for path, digest in corpus_hashes.items()
    ):
        raise ArtifactValidationError("invalid corpus document hash metadata")

    summary = _require_mapping(artifact, "summary", path="artifact")
    results = _require_mapping(artifact, "results", path="artifact")
    expected_query_ids: List[str] | None = None
    for method in EXPECTED_METHODS:
        method_summary = _require_mapping(summary, method, path="artifact.summary")
        summary_metrics = _require_mapping(
            method_summary, "metrics_by_k", path=f"artifact.summary.{method}"
        )
        rows = results.get(method)
        if not isinstance(rows, list) or len(rows) != query_count:
            raise ArtifactValidationError(
                f"results.{method} must contain {query_count} query rows"
            )

        query_ids = []
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise ArtifactValidationError(
                    f"results.{method}[{row_number}] must be an object"
                )
            query_id = _require_non_empty_string(
                row, "query_id", path=f"results.{method}[{row_number}]"
            )
            query_ids.append(query_id)
            if row.get("method") != method:
                raise ArtifactValidationError(
                    f"results.{method}[{row_number}].method mismatch"
                )
            _require_non_empty_string(
                row, "query", path=f"results.{method}[{row_number}]"
            )
            if not isinstance(row.get("retrieved_chunk_ids"), list):
                raise ArtifactValidationError(
                    f"results.{method}[{row_number}] missing retrieved_chunk_ids"
                )
            if not isinstance(row.get("relevant_document_labels"), list):
                raise ArtifactValidationError(
                    f"results.{method}[{row_number}] missing relevance labels"
                )

            metrics_status = row.get("metrics_status")
            metrics_by_k = row.get("metrics_by_k")
            if metrics_status == "evaluated":
                if not isinstance(metrics_by_k, Mapping):
                    raise ArtifactValidationError(
                        f"results.{method}[{row_number}] missing metrics_by_k"
                    )
                for k in k_values:
                    metrics = metrics_by_k.get(str(k))
                    if not isinstance(metrics, Mapping) or any(
                        field not in metrics
                        for field in (
                            "hit",
                            "recall",
                            "reciprocal_rank",
                            "first_relevant_rank",
                        )
                    ):
                        raise ArtifactValidationError(
                            f"results.{method}[{row_number}] missing metrics at K={k}"
                        )
            elif metrics_status != "not_applicable_no_relevant_documents":
                raise ArtifactValidationError(
                    f"results.{method}[{row_number}] has invalid metrics_status"
                )

        if len(set(query_ids)) != len(query_ids):
            raise ArtifactValidationError(f"results.{method} has duplicate query IDs")
        if expected_query_ids is None:
            expected_query_ids = query_ids
        elif query_ids != expected_query_ids:
            raise ArtifactValidationError("methods do not contain identical query IDs")

        for k in k_values:
            if not isinstance(summary_metrics.get(str(k)), Mapping):
                raise ArtifactValidationError(
                    f"summary.{method} missing metrics at K={k}"
                )

    comparisons = _require_mapping(
        artifact, "comparisons_by_k", path="artifact"
    )
    for k in k_values:
        if not isinstance(comparisons.get(str(k)), Mapping):
            raise ArtifactValidationError(f"comparisons_by_k missing K={k}")

    if project_root is not None:
        dataset_path = _project_file(project_root, dataset_source)
        if not dataset_path.exists() or _sha256(dataset_path) != dataset_hash:
            raise ArtifactValidationError("dataset hash does not match source artifact")
        for recorded_path, recorded_hash in corpus_hashes.items():
            corpus_path = _project_file(project_root, recorded_path)
            if not corpus_path.exists() or _sha256(corpus_path) != recorded_hash:
                raise ArtifactValidationError(
                    f"corpus hash does not match source artifact: {recorded_path}"
                )

    warnings = []
    if artifact.get("artifact_version") is None:
        warnings.append("W6-T4 source artifact does not declare artifact_version")
    return warnings


def _index_results(
    artifact: Mapping[str, Any],
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    return {
        method: {row["query_id"]: row for row in artifact["results"][method]}
        for method in EXPECTED_METHODS
    }


def _quality(result: Mapping[str, Any], k: int) -> tuple[float, float]:
    metrics = result["metrics_by_k"][str(k)]
    return float(metrics["recall"]), float(metrics["reciprocal_rank"])


def is_partial_multi_relevant(result: Mapping[str, Any], *, k: int) -> bool:
    relevant = result.get("relevant_document_labels", [])
    if not isinstance(relevant, list) or len(set(relevant)) <= 1:
        return False
    metrics = result["metrics_by_k"][str(k)]
    return (
        bool(metrics["hit"])
        and float(metrics["reciprocal_rank"]) > 0
        and 0 < float(metrics["recall"]) < 1
    )


def build_outcome_groups(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    indexed = _index_results(artifact)
    groups_by_k: Dict[str, Any] = {}

    for k in artifact["configuration"]["k_values"]:
        groups: Dict[str, List[Any]] = {
            "bm25_better_than_vector": [],
            "vector_better_than_bm25": [],
            "hybrid_better_than_vector": [],
            "hybrid_worse_than_vector": [],
            "hybrid_equal_to_vector": [],
            "bm25_better_than_hybrid": [],
            "hybrid_better_than_bm25": [],
            "bm25_equal_to_hybrid": [],
            "all_methods_succeed": [],
            "all_methods_miss": [],
            "multi_relevant_partially_recalled": [],
        }
        for query_id, vector in indexed["vector"].items():
            if vector["metrics_status"] != "evaluated":
                continue
            bm25 = indexed["bm25"][query_id]
            hybrid = indexed["hybrid_rrf"][query_id]
            vector_quality = _quality(vector, k)
            bm25_quality = _quality(bm25, k)
            hybrid_quality = _quality(hybrid, k)

            if bm25_quality > vector_quality:
                groups["bm25_better_than_vector"].append(query_id)
            elif vector_quality > bm25_quality:
                groups["vector_better_than_bm25"].append(query_id)

            if hybrid_quality > vector_quality:
                groups["hybrid_better_than_vector"].append(query_id)
            elif hybrid_quality < vector_quality:
                groups["hybrid_worse_than_vector"].append(query_id)
            else:
                groups["hybrid_equal_to_vector"].append(query_id)

            if bm25_quality > hybrid_quality:
                groups["bm25_better_than_hybrid"].append(query_id)
            elif hybrid_quality > bm25_quality:
                groups["hybrid_better_than_bm25"].append(query_id)
            else:
                groups["bm25_equal_to_hybrid"].append(query_id)

            qualities = (vector_quality, bm25_quality, hybrid_quality)
            if all(recall == 1.0 for recall, _ in qualities):
                groups["all_methods_succeed"].append(query_id)
            if all(quality == (0.0, 0.0) for quality in qualities):
                groups["all_methods_miss"].append(query_id)

            partial_methods = [
                method
                for method, result in (
                    ("vector", vector),
                    ("bm25", bm25),
                    ("hybrid_rrf", hybrid),
                )
                if is_partial_multi_relevant(result, k=k)
            ]
            if partial_methods:
                groups["multi_relevant_partially_recalled"].append(
                    {"query_id": query_id, "methods": partial_methods}
                )

        groups_by_k[str(k)] = {
            "groups": groups,
            "empty_groups": [name for name, values in groups.items() if not values],
        }
    return groups_by_k


def _rank(identity: str, ranked_ids: Sequence[str]) -> int | None:
    return next(
        (rank for rank, candidate in enumerate(ranked_ids, start=1) if candidate == identity),
        None,
    )


def build_rank_movements(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    indexed = _index_results(artifact)
    k_values = artifact["configuration"]["k_values"]
    items = []
    for query_id, vector in indexed["vector"].items():
        relevant_chunk_ids = vector.get("relevant_chunk_ids")
        if not relevant_chunk_ids:
            continue
        for chunk_id in relevant_chunk_ids:
            ranks = {
                method: _rank(
                    chunk_id,
                    indexed[method][query_id]["retrieved_chunk_ids"],
                )
                for method in EXPECTED_METHODS
            }
            comparisons = {}
            for name, source, target in (
                ("vector_to_bm25", "vector", "bm25"),
                ("vector_to_hybrid", "vector", "hybrid_rrf"),
                ("bm25_to_hybrid", "bm25", "hybrid_rrf"),
            ):
                comparisons[name] = {
                    str(k): classify_rank_movement(
                        ranks[source],
                        ranks[target],
                        k=k,
                    )
                    for k in k_values
                }
            items.append(
                {
                    "query_id": query_id,
                    "chunk_id": chunk_id,
                    "ranks": ranks,
                    "comparisons_by_k": comparisons,
                }
            )
    return {
        "scope": "strict chunk-labelled queries only",
        "query_count": len({item["query_id"] for item in items}),
        "relevant_chunk_count": len(items),
        "items": items,
    }


def build_query_mechanism_annotations(
    artifact: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Use only explicit answerability and relevance cardinality; do not infer semantics."""
    rows = artifact["results"]["vector"]
    annotations = []
    for row in rows:
        relevant = row["relevant_document_labels"]
        if not row["should_answer"]:
            assigned_type = "no_answer"
            reason = "Dataset explicitly marks should_answer=false with no relevant documents."
            evidence = {"should_answer": False, "relevant_document_count": 0}
            confidence = "high"
            source = "W6-T4 fields derived from the original dataset"
        elif len(set(relevant)) > 1:
            assigned_type = "multi_document"
            reason = "The Ground Truth explicitly requires more than one relevant document."
            evidence = {
                "relevant_document_count": len(set(relevant)),
                "relevant_document_labels": relevant,
            }
            confidence = "high"
            source = "W6-T4 relevance labels"
        else:
            assigned_type = "unclassified"
            reason = (
                "The dataset has no independent retrieval-mechanism label for this query."
            )
            evidence = {"dataset_category": row["category"]}
            confidence = "not_applicable"
            source = "analysis boundary; no mechanism annotation available"
        annotations.append(
            {
                "query_id": row["query_id"],
                "assigned_type": assigned_type,
                "reason": reason,
                "evidence": evidence,
                "confidence": confidence,
                "review_status": "deterministic_from_recorded_fields",
                "annotation_source": source,
            }
        )
    return annotations


def build_failure_stages(
    artifact: Mapping[str, Any],
    rank_movements: Mapping[str, Any],
) -> Dict[str, Any]:
    indexed = _index_results(artifact)
    k_values = artifact["configuration"]["k_values"]
    max_k = max(k_values)

    source_recall_gaps = []
    ranking_cutoff_failures = []
    multi_relevant_failures = []
    for method in EXPECTED_METHODS:
        for query_id, row in indexed[method].items():
            if row["metrics_status"] != "evaluated":
                continue
            if float(row["metrics_by_k"][str(max_k)]["recall"]) < 1.0:
                source_recall_gaps.append({"query_id": query_id, "method": method})
            for k in k_values:
                recall_at_k = float(row["metrics_by_k"][str(k)]["recall"])
                recall_at_depth = float(row["metrics_by_k"][str(max_k)]["recall"])
                if recall_at_k < recall_at_depth:
                    ranking_cutoff_failures.append(
                        {"query_id": query_id, "method": method, "k": k}
                    )
                if is_partial_multi_relevant(row, k=k):
                    multi_relevant_failures.append(
                        {"query_id": query_id, "method": method, "k": k}
                    )

    fusion_did_not_cross = []
    for item in rank_movements["items"]:
        for k, movement in item["comparisons_by_k"]["vector_to_hybrid"].items():
            if movement["cutoff_effect"] == "moved_up_but_did_not_cross_cutoff":
                fusion_did_not_cross.append(
                    {
                        "query_id": item["query_id"],
                        "chunk_id": item["chunk_id"],
                        "k": int(k),
                        "vector_rank": item["ranks"]["vector"],
                        "hybrid_rank": item["ranks"]["hybrid_rrf"],
                    }
                )

    summary = artifact["summary"]
    saturation_points = []
    for method in EXPECTED_METHODS:
        for k in k_values:
            metrics = summary[method]["metrics_by_k"][str(k)]
            if float(metrics["hit_rate"]) == 1.0 and float(metrics["mrr"]) == 1.0:
                saturation_points.append({"method": method, "k": k})

    return {
        "source_recall_gap": {
            "status": "observed" if source_recall_gaps else "not_observed",
            "evidence": source_recall_gaps,
            "scope": f"document relevance within recorded depth {max_k}",
        },
        "ranking_cutoff_failure": {
            "status": "observed" if ranking_cutoff_failures else "not_observed",
            "evidence": ranking_cutoff_failures,
        },
        "fusion_did_not_cross_cutoff": {
            "status": "observed" if fusion_did_not_cross else "not_observed",
            "evidence": fusion_did_not_cross,
        },
        "multi_relevant_completeness_failure": {
            "status": "observed" if multi_relevant_failures else "not_observed",
            "evidence": multi_relevant_failures,
        },
        "metric_saturation": {
            "status": "observed",
            "evidence": saturation_points,
            "explanation": "Hit Rate and MRR are 1.0 for every method and K.",
        },
        "label_granularity_limitation": {
            "status": "observed",
            "evidence": {
                "primary_label_type": artifact["ground_truth"]["label_type"],
                "strict_chunk_label_coverage": artifact["ground_truth"][
                    "strict_chunk_label_coverage"
                ],
            },
        },
        "dataset_coverage_limitation": {
            "status": "observed",
            "evidence": {
                "document_count": len(artifact["configuration"]["indexed_documents"]),
                "chunk_count": artifact["configuration"]["indexed_chunk_count"],
                "query_count": artifact["configuration"]["query_count"],
                "mechanism_labels_available": False,
            },
        },
        "unanswerable_evaluation_gap": {
            "status": "observed",
            "evidence": {
                "query_count": sum(
                    1
                    for row in artifact["results"]["vector"]
                    if not row["should_answer"]
                ),
                "core_metrics_status": "not_applicable_no_relevant_documents",
            },
        },
    }


def _representative_q027(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    indexed = _index_results(artifact)
    query_id = "q027"
    vector = indexed["vector"][query_id]
    relevant_chunks = vector.get("relevant_chunk_ids") or []
    return {
        "query_id": query_id,
        "query": vector["query"],
        "dataset_category": vector["category"],
        "analysis_mechanism_type": "multi_document",
        "relevant_document_labels": vector["relevant_document_labels"],
        "relevant_chunk_ids": relevant_chunks,
        "methods": {
            method: {
                "retrieved_chunk_ids": indexed[method][query_id]["retrieved_chunk_ids"],
                "strict_relevant_chunk_ranks": {
                    chunk_id: _rank(
                        chunk_id,
                        indexed[method][query_id]["retrieved_chunk_ids"],
                    )
                    for chunk_id in relevant_chunks
                },
                "metrics_at_3": indexed[method][query_id]["metrics_by_k"]["3"],
            }
            for method in EXPECTED_METHODS
        },
        "observed_stage": [
            "ranking_cutoff_failure",
            "fusion_did_not_cross_cutoff",
            "multi_relevant_completeness_failure",
            "metric_saturation",
        ],
        "causal_attribution": (
            "insufficient evidence: the artifact stores rankings and scores, not per-term BM25 contributions or model reasoning"
        ),
    }


def analyze_artifact(
    artifact: Mapping[str, Any],
    *,
    source_artifact: str,
    validation_warnings: Sequence[str],
) -> Dict[str, Any]:
    outcome_groups = build_outcome_groups(artifact)
    rank_movements = build_rank_movements(artifact)
    failure_stages = build_failure_stages(artifact, rank_movements)
    annotations = build_query_mechanism_annotations(artifact)

    def group_by_k(name: str) -> Dict[str, List[str]]:
        return {
            k: list(grouped["groups"][name])
            for k, grouped in outcome_groups.items()
        }

    hybrid_equal_by_k = group_by_k("hybrid_equal_to_vector")
    evaluated_query_count = artifact["summary"]["vector"]["evaluated_query_count"]
    hybrid_equal_at_every_k = all(
        len(query_ids) == evaluated_query_count
        for query_ids in hybrid_equal_by_k.values()
    )

    return {
        "task": "W6-T5",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_generated_at": datetime.now(timezone.utc).isoformat(),
        "source_evidence": {
            "artifact_path": source_artifact,
            "artifact_task": artifact["task"],
            "artifact_version": artifact.get("artifact_version"),
            "artifact_generated_at": artifact["generated_at"],
            "dataset_source": artifact["ground_truth"]["source"],
            "dataset_sha256": artifact["ground_truth"]["dataset_sha256"],
            "corpus_document_sha256": artifact["configuration"][
                "bootstrap_document_sha256"
            ],
            "validation_status": "passed_with_warnings" if validation_warnings else "passed",
            "validation_warnings": list(validation_warnings),
        },
        "experiment_conditions": {
            "document_count": len(artifact["configuration"]["indexed_documents"]),
            "chunk_count": artifact["configuration"]["indexed_chunk_count"],
            "query_count": artifact["configuration"]["query_count"],
            "answerable_query_count": sum(
                1 for row in artifact["results"]["vector"] if row["should_answer"]
            ),
            "unanswerable_query_count": sum(
                1 for row in artifact["results"]["vector"] if not row["should_answer"]
            ),
            "primary_ground_truth": artifact["ground_truth"]["label_type"],
            "strict_chunk_label_coverage": artifact["ground_truth"][
                "strict_chunk_label_coverage"
            ],
            "methods": artifact["configuration"]["methods"],
            "k_values": artifact["configuration"]["k_values"],
            "hybrid_candidate_depth": artifact["configuration"][
                "hybrid_candidate_depth"
            ],
            "rrf_k": artifact["configuration"]["rrf_k"],
        },
        "comparison_definition": (
            "Compare (Recall@K, reciprocal rank@K) lexicographically, matching W6-T4."
        ),
        "outcome_groups_by_k": outcome_groups,
        "query_mechanism_annotations": annotations,
        "rank_movements": rank_movements,
        "failure_stages": failure_stages,
        "representative_cases": {"q027": _representative_q027(artifact)},
        "observed_findings": {
            "bm25_strict_advantage_by_k": group_by_k("bm25_better_than_vector"),
            "vector_strict_advantage_by_k": group_by_k("vector_better_than_bm25"),
            "hybrid_strict_advantage_over_vector_by_k": group_by_k(
                "hybrid_better_than_vector"
            ),
            "hybrid_strict_regression_vs_vector_by_k": group_by_k(
                "hybrid_worse_than_vector"
            ),
            "all_methods_miss_by_k": group_by_k("all_methods_miss"),
            "hybrid_equal_to_vector_by_k": hybrid_equal_by_k,
            "hybrid_aggregate_relationship": (
                "equal_to_vector_at_every_evaluated_k"
                if hybrid_equal_at_every_k
                else "not_equal_to_vector_at_every_evaluated_k"
            ),
        },
        "evidence_limitations": [
            "Primary relevance is document-filename level, not full strict chunk relevance.",
            "Only three queries have strict chunk labels.",
            "The corpus has four documents and twelve chunks.",
            "Twenty-two of twenty-three answerable queries have one relevant document.",
            "No independent exact-term, identifier, abbreviation, or semantic-paraphrase labels exist.",
            "Four unanswerable queries are outside ordinary Hit/Recall/MRR macro-averages.",
            "The source artifact does not declare an explicit artifact version.",
        ],
        "unverified_hypotheses": [
            {
                "status": "unverified",
                "hypothesis": "A larger semantic-paraphrase subset may expose Vector advantages not present here.",
            },
            {
                "status": "unverified",
                "hypothesis": "Corpus size or lexical overlap may contribute to the absence of Vector-better cases.",
            },
            {
                "status": "unverified",
                "hypothesis": "A content-aware reranker may move q027's HR chunk into top 3, but improvement is not guaranteed.",
            },
            {
                "status": "unverified",
                "hypothesis": "Different fusion parameters could change ranks; W6-T5 does not tune or test them.",
            },
        ],
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze recorded W6-T4 retrieval failures without rerunning retrieval."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_path = _resolve(args.source)
    output_path = _resolve(args.output)
    artifact = load_artifact(source_path)
    warnings = validate_artifact(artifact, project_root=PROJECT_ROOT)
    try:
        source_label = str(source_path.relative_to(PROJECT_ROOT))
    except ValueError:
        source_label = str(source_path)
    analysis = analyze_artifact(
        artifact,
        source_artifact=source_label,
        validation_warnings=warnings,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(analysis["observed_findings"], ensure_ascii=False, indent=2))
    print(f"Wrote failure analysis to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
