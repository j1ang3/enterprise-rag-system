"""Timing and aggregation primitives for the frozen W7-T4 benchmark."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean, median
from time import perf_counter
from typing import Any

from app.evaluation.retrieval_comparison import METHOD_NAMES


PHASES = ("cold_start", "warmup", "measured")
SUCCESS = "success"
FAILED = "failed"


def measure_query_methods(
    query_id: str,
    query: str,
    *,
    run_id: str,
    phase: str,
    vector_retriever: Any,
    hybrid_retriever: Any,
    reranker: Any,
    config: Mapping[str, Any],
    clock: Callable[[], float] = perf_counter,
) -> list[dict[str, Any]]:
    """Measure the three frozen methods while sharing one Hybrid candidate list."""
    if not query_id.strip():
        raise ValueError("query_id must not be empty")
    if not query.strip():
        raise ValueError("query must not be empty")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")

    evaluation_depth = int(config["offline_evaluation_depth"])
    records: list[dict[str, Any]] = []

    vector_results, vector_ms, vector_error = _measure_call(
        lambda: vector_retriever.retrieve(query, evaluation_depth), clock=clock
    )
    if vector_error is None:
        _validate_ranked_results(
            vector_results, method="vector", max_count=evaluation_depth
        )
        records.append(
            _success_record(
                query_id=query_id,
                method="vector",
                run_id=run_id,
                phase=phase,
                total_retrieval_ms=vector_ms,
                stage_latency_ms={"vector_ms": vector_ms},
                result_chunk_ids=_chunk_ids(vector_results),
            )
        )
    else:
        records.append(
            _failure_record(
                query_id=query_id,
                method="vector",
                run_id=run_id,
                phase=phase,
                elapsed_before_failure_ms=vector_ms,
                error=vector_error,
            )
        )

    hybrid_results, hybrid_ms, hybrid_error = _measure_call(
        lambda: hybrid_retriever.retrieve_fused(
            query,
            int(config["rrf_output_count"]),
            candidate_depth=int(config["per_source_candidate_depth"]),
            rrf_k=int(config["rrf_k"]),
        ),
        clock=clock,
    )
    if hybrid_error is not None:
        records.append(
            _failure_record(
                query_id=query_id,
                method="hybrid_rrf",
                run_id=run_id,
                phase=phase,
                elapsed_before_failure_ms=hybrid_ms,
                error=hybrid_error,
            )
        )
        records.append(
            _failure_record(
                query_id=query_id,
                method="hybrid_reranker",
                run_id=run_id,
                phase=phase,
                elapsed_before_failure_ms=hybrid_ms,
                error=RuntimeError(
                    "reranking was not attempted because Hybrid retrieval failed"
                ),
            )
        )
        return records

    _validate_ranked_results(
        hybrid_results, method="hybrid_rrf", max_count=evaluation_depth
    )
    hybrid_ids = _chunk_ids(hybrid_results)
    records.append(
        _success_record(
            query_id=query_id,
            method="hybrid_rrf",
            run_id=run_id,
            phase=phase,
            total_retrieval_ms=hybrid_ms,
            stage_latency_ms={"hybrid_retrieval_and_fusion_ms": hybrid_ms},
            result_chunk_ids=hybrid_ids,
        )
    )

    reranked_results, rerank_ms, rerank_error = _measure_call(
        lambda: reranker.rerank(
            query,
            hybrid_results,
            top_k=evaluation_depth,
        ),
        clock=clock,
    )
    if rerank_error is not None:
        records.append(
            _failure_record(
                query_id=query_id,
                method="hybrid_reranker",
                run_id=run_id,
                phase=phase,
                elapsed_before_failure_ms=hybrid_ms + rerank_ms,
                error=rerank_error,
                stage_latency_ms={
                    "hybrid_retrieval_and_fusion_ms": hybrid_ms,
                    "rerank_elapsed_before_failure_ms": rerank_ms,
                },
            )
        )
        return records

    _validate_ranked_results(
        reranked_results, method="hybrid_reranker", max_count=evaluation_depth
    )
    reranked_ids = _chunk_ids(reranked_results)
    if len(hybrid_ids) != len(reranked_ids) or set(hybrid_ids) != set(
        reranked_ids
    ):
        raise RuntimeError("reranker output is not a permutation of Hybrid candidates")
    records.append(
        _success_record(
            query_id=query_id,
            method="hybrid_reranker",
            run_id=run_id,
            phase=phase,
            total_retrieval_ms=hybrid_ms + rerank_ms,
            rerank_ms=rerank_ms,
            stage_latency_ms={
                "shared_hybrid_retrieval_and_fusion_ms": hybrid_ms,
                "rerank_ms": rerank_ms,
            },
            result_chunk_ids=reranked_ids,
            pre_rerank_chunk_ids=hybrid_ids,
        )
    )
    return records


def aggregate_latency_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate measured samples only; cold-start and warm-up stay excluded."""
    for sample in samples:
        validate_latency_record(sample)

    aggregates: dict[str, dict[str, Any]] = {}
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    for method in METHOD_NAMES:
        method_samples = [sample for sample in measured if sample["method"] == method]
        successful = [
            sample
            for sample in method_samples
            if sample["status"] == SUCCESS
        ]
        total_values = [float(sample["total_retrieval_ms"]) for sample in successful]
        aggregates[method] = {
            **_describe(total_values),
            "attempted_sample_count": len(method_samples),
            "failed_sample_count": len(method_samples) - len(successful),
        }
    return aggregates


def aggregate_rerank_stage(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the isolated warm rerank stage from successful measured samples."""
    values = [
        float(sample["rerank_ms"])
        for sample in samples
        if sample["phase"] == "measured"
        and sample["method"] == "hybrid_reranker"
        and sample["status"] == SUCCESS
    ]
    return _describe(values)


def aggregate_latency_by_query(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Retain per-query distributions without creating a separate analytics system."""
    query_ids = sorted(
        {
            str(sample["query_id"])
            for sample in samples
            if sample["phase"] == "measured"
        }
    )
    return {
        query_id: {
            method: _describe(
                [
                    float(sample["total_retrieval_ms"])
                    for sample in samples
                    if sample["phase"] == "measured"
                    and sample["query_id"] == query_id
                    and sample["method"] == method
                    and sample["status"] == SUCCESS
                ]
            )
            for method in METHOD_NAMES
        }
        for query_id in query_ids
    }


def calculate_latency_delta(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> dict[str, dict[str, float | None]]:
    """Calculate absolute and relative aggregate deltas without dividing by zero."""
    deltas: dict[str, dict[str, float | None]] = {}
    for statistic in ("mean_ms", "p50_ms", "p95_ms"):
        baseline_value = baseline.get(statistic)
        challenger_value = challenger.get(statistic)
        if baseline_value is None or challenger_value is None:
            deltas[statistic] = {
                "absolute_delta_ms": None,
                "relative_delta_percent": None,
            }
            continue
        absolute = float(challenger_value) - float(baseline_value)
        relative = (
            absolute / float(baseline_value) * 100.0
            if abs(float(baseline_value)) > 1e-12
            else None
        )
        deltas[statistic] = {
            "absolute_delta_ms": absolute,
            "relative_delta_percent": relative,
        }
    return deltas


def validate_quality_artifact(
    artifact: Mapping[str, Any],
    *,
    frozen_manifest: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> None:
    """Prove that W7-T4 consumes the frozen W7-T3 quality evidence unchanged."""
    if artifact.get("task") != "W7-T3":
        raise ValueError("quality artifact task must be W7-T3")
    if tuple(artifact.get("per_query_results", {})) != METHOD_NAMES:
        raise ValueError("quality artifact methods differ from W7-T3")
    if tuple(artifact.get("aggregate_results", {})) != METHOD_NAMES:
        raise ValueError("quality aggregate methods differ from W7-T3")
    recorded_manifest = artifact.get("frozen_manifest", {})
    if recorded_manifest.get("sha256") != frozen_manifest_sha256:
        raise ValueError("quality artifact frozen manifest hash differs")
    if artifact.get("resolved_configuration") != frozen_manifest.get(
        "resolved_configuration"
    ):
        raise ValueError("quality artifact configuration differs from frozen manifest")
    dataset = artifact.get("dataset_identity", {})
    manifest_dataset = frozen_manifest.get("dataset", {})
    for field in ("path", "sha256"):
        if dataset.get(field) != manifest_dataset.get(field):
            raise ValueError(f"quality artifact dataset {field} differs")
    expected_query_ids = frozen_manifest.get("split", {}).get(
        "evaluation_query_ids"
    )
    if dataset.get("evaluation_query_ids") != expected_query_ids:
        raise ValueError("quality artifact query IDs differ from frozen manifest")
    if artifact.get("corpus_identity") != frozen_manifest.get("corpus"):
        raise ValueError("quality artifact corpus differs from frozen manifest")


def extract_quality_evidence(
    artifact: Mapping[str, Any],
    *,
    primary_k: int,
) -> dict[str, Any]:
    """Copy verified W7-T3 metrics and calculate transparent per-metric deltas."""
    aggregates = artifact["aggregate_results"]
    metrics_by_method = {
        method: aggregates[method]["metrics_by_k"] for method in METHOD_NAMES
    }
    key = str(primary_k)
    primary = {method: metrics_by_method[method][key] for method in METHOD_NAMES}
    return {
        "source_task": "W7-T3",
        "primary_k": primary_k,
        "metrics_by_method_and_k": metrics_by_method,
        "primary_metrics_by_method": primary,
        "quality_deltas": {
            "vector_to_hybrid": _metric_delta(
                primary["vector"], primary["hybrid_rrf"]
            ),
            "hybrid_to_hybrid_reranker": _metric_delta(
                primary["hybrid_rrf"], primary["hybrid_reranker"]
            ),
        },
        "reranker_effect_summary": artifact["reranker_effect"]["summary"],
    }


def validate_latency_record(record: Mapping[str, Any]) -> None:
    required = {
        "query_id",
        "method",
        "run_id",
        "phase",
        "status",
        "total_retrieval_ms",
        "rerank_ms",
        "stage_latency_ms",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"latency record is missing fields: {sorted(missing)}")
    if record["method"] not in METHOD_NAMES:
        raise ValueError(f"unknown latency method: {record['method']}")
    if record["phase"] not in PHASES:
        raise ValueError(f"unknown latency phase: {record['phase']}")
    if record["status"] not in (SUCCESS, FAILED):
        raise ValueError(f"unknown latency status: {record['status']}")
    if record["status"] == SUCCESS:
        duration = record["total_retrieval_ms"]
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError("successful latency record requires a numeric duration")
        if not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError("successful latency duration must be positive and finite")
    elif record["total_retrieval_ms"] is not None:
        raise ValueError("failed latency record must not report a successful duration")


def _measure_call(
    operation: Callable[[], Any],
    *,
    clock: Callable[[], float],
) -> tuple[Any | None, float, Exception | None]:
    started = clock()
    try:
        result = operation()
    except Exception as exc:
        duration_ms = (clock() - started) * 1000.0
        return None, duration_ms, exc
    duration_ms = (clock() - started) * 1000.0
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise RuntimeError("measured duration must be positive and finite")
    return result, duration_ms, None


def _success_record(
    *,
    query_id: str,
    method: str,
    run_id: str,
    phase: str,
    total_retrieval_ms: float,
    stage_latency_ms: Mapping[str, float],
    result_chunk_ids: Sequence[str],
    rerank_ms: float | None = None,
    pre_rerank_chunk_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    record = {
        "query_id": query_id,
        "method": method,
        "run_id": run_id,
        "phase": phase,
        "status": SUCCESS,
        "total_retrieval_ms": total_retrieval_ms,
        "rerank_ms": rerank_ms,
        "stage_latency_ms": dict(stage_latency_ms),
        "result_chunk_ids": list(result_chunk_ids),
        "pre_rerank_chunk_ids": (
            list(pre_rerank_chunk_ids) if pre_rerank_chunk_ids is not None else None
        ),
        "error": None,
    }
    validate_latency_record(record)
    return record


def _failure_record(
    *,
    query_id: str,
    method: str,
    run_id: str,
    phase: str,
    elapsed_before_failure_ms: float,
    error: Exception,
    stage_latency_ms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    record = {
        "query_id": query_id,
        "method": method,
        "run_id": run_id,
        "phase": phase,
        "status": FAILED,
        "total_retrieval_ms": None,
        "rerank_ms": None,
        "stage_latency_ms": dict(stage_latency_ms or {}),
        "result_chunk_ids": None,
        "pre_rerank_chunk_ids": None,
        "elapsed_before_failure_ms": elapsed_before_failure_ms,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    validate_latency_record(record)
    return record


def _validate_ranked_results(
    results: Any,
    *,
    method: str,
    max_count: int,
) -> None:
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ValueError(f"{method} results must be a sequence")
    if len(results) > max_count:
        raise RuntimeError(f"{method} returned more than the frozen output depth")
    chunk_ids = _chunk_ids(results)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"{method} returned duplicate chunk identities")


def _chunk_ids(results: Sequence[Mapping[str, Any]]) -> list[str]:
    chunk_ids = []
    for rank, result in enumerate(results, start=1):
        chunk_id = result.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"result rank {rank} is missing chunk_id")
        chunk_ids.append(chunk_id)
    return chunk_ids


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    normalized = [float(value) for value in values]
    return {
        "sample_count": len(normalized),
        "mean_ms": fmean(normalized),
        "p50_ms": median(normalized),
        "p95_ms": _percentile(normalized, 0.95),
        "min_ms": min(normalized),
        "max_ms": max(normalized),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Use the common linear interpolation between adjacent sorted samples."""
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * weight


def _metric_delta(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> dict[str, float]:
    return {
        metric: float(challenger[metric]) - float(baseline[metric])
        for metric in ("hit_rate", "recall", "mrr")
    }
