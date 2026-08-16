from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.latency import (  # noqa: E402
    FAILED,
    aggregate_latency_by_query,
    aggregate_latency_samples,
    aggregate_rerank_stage,
    calculate_latency_delta,
    extract_quality_evidence,
    measure_query_methods,
    validate_quality_artifact,
)
from app.evaluation.retrieval_comparison import METHOD_NAMES  # noqa: E402
from scripts.evaluate_reranked_retrieval import (  # noqa: E402
    build_method_components,
    validate_frozen_manifest,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "retrieval_evaluation_config.json"
DEFAULT_QUALITY_ARTIFACT = (
    PROJECT_ROOT / "evals" / "results" / "W7-T3-retrieval-evaluation.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evals" / "results" / "W7-T4-latency-trade-off.json"
)
PROTECTED_INPUTS = {
    path.resolve()
    for path in (
        PROJECT_ROOT / "evals" / "results" / "W6-T4-retrieval-evaluation.json",
        PROJECT_ROOT / "evals" / "results" / "W6-T5-failure-analysis.json",
        PROJECT_ROOT
        / "evals"
        / "results"
        / "W7-T2-retrieval-configuration.json",
        DEFAULT_QUALITY_ARTIFACT,
        DEFAULT_MANIFEST,
    )
}


def run_benchmark(
    *,
    manifest_path: Path,
    quality_artifact_path: Path,
    output_path: Path,
    measured_pass_count: int = 5,
    overwrite: bool = False,
) -> dict[str, Any]:
    if measured_pass_count <= 0:
        raise ValueError("measured_pass_count must be positive")

    resolved = validate_frozen_manifest(manifest_path)
    quality_artifact = _load_json(quality_artifact_path, name="W7-T3 artifact")
    validate_quality_artifact(
        quality_artifact,
        frozen_manifest=resolved["manifest"],
        frozen_manifest_sha256=resolved["manifest_sha256"],
    )

    config = resolved["config"]
    component_started = perf_counter()
    vector, hybrid, reranker = build_method_components(
        chunk_index_path=resolved["chunk_index_path"],
        vector_index_path=resolved["vector_index_path"],
        vector_min_score=float(config["vector_min_score"]),
    )
    component_construction_ms = (perf_counter() - component_started) * 1000.0
    _validate_runtime_components(hybrid, reranker, config=config)

    cold_query_id = resolved["split"]["tuning_query_ids"][0]
    cold_query = str(resolved["examples_by_id"][cold_query_id]["question"])
    cold_start_samples = measure_query_methods(
        cold_query_id,
        cold_query,
        run_id="cold-start-1",
        phase="cold_start",
        vector_retriever=vector,
        hybrid_retriever=hybrid,
        reranker=reranker,
        config=config,
    )
    actual_reranker_device = _actual_reranker_device(reranker)

    warmup_samples: list[dict[str, Any]] = []
    for query_id in resolved["evaluation_query_ids"]:
        warmup_samples.extend(
            measure_query_methods(
                query_id,
                str(resolved["examples_by_id"][query_id]["question"]),
                run_id="warmup-pass-1",
                phase="warmup",
                vector_retriever=vector,
                hybrid_retriever=hybrid,
                reranker=reranker,
                config=config,
            )
        )

    measured_samples: list[dict[str, Any]] = []
    for pass_number in range(1, measured_pass_count + 1):
        run_id = f"measured-pass-{pass_number}"
        for query_id in resolved["evaluation_query_ids"]:
            measured_samples.extend(
                measure_query_methods(
                    query_id,
                    str(resolved["examples_by_id"][query_id]["question"]),
                    run_id=run_id,
                    phase="measured",
                    vector_retriever=vector,
                    hybrid_retriever=hybrid,
                    reranker=reranker,
                    config=config,
                )
            )

    _validate_measured_rankings(
        measured_samples,
        quality_artifact=quality_artifact,
    )
    all_samples = cold_start_samples + warmup_samples + measured_samples
    aggregate_results = aggregate_latency_samples(all_samples)
    rerank_stage = aggregate_rerank_stage(all_samples)
    latency_deltas = {
        "vector_to_hybrid": calculate_latency_delta(
            aggregate_results["vector"], aggregate_results["hybrid_rrf"]
        ),
        "hybrid_to_hybrid_reranker": calculate_latency_delta(
            aggregate_results["hybrid_rrf"],
            aggregate_results["hybrid_reranker"],
        ),
    }
    quality_evidence = extract_quality_evidence(
        quality_artifact,
        primary_k=int(config["operational_final_top_k"]),
    )
    failure_count = sum(
        sample["status"] == FAILED for sample in measured_samples
    )

    payload = {
        "artifact_version": 1,
        "task": "W7-T4",
        "experiment_id": "W7-T4-frozen-retrieval-latency-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if failure_count == 0 else "incomplete",
        "scope": (
            "Controlled sequential retrieval/ranking latency only: Vector vs "
            "Hybrid RRF vs Hybrid RRF + fixed Reranker. No LLM generation, "
            "concurrency, throughput testing, or tuning."
        ),
        "repository_state": _repository_state(),
        "source_artifacts": {
            "frozen_manifest": _artifact_reference(manifest_path),
            "w7_t2": resolved["manifest"]["source_w7_t2_artifact"],
            "w7_t3_quality": _artifact_reference(quality_artifact_path),
        },
        "dataset_identity": {
            **resolved["manifest"]["dataset"],
            "benchmark_query_ids": resolved["evaluation_query_ids"],
            "benchmark_query_count": len(resolved["evaluation_query_ids"]),
        },
        "corpus_identity": resolved["manifest"]["corpus"],
        "configuration_identity": {
            "selected_w7_t2_config_id": config["selected_w7_t2_config_id"],
            "resolved_configuration": config,
            "configuration_changed_for_latency": False,
        },
        "method_identity": {
            "methods": list(METHOD_NAMES),
            "vector": "VectorRetriever -> ranked results",
            "hybrid_rrf": (
                "VectorRetriever + BM25Retriever -> RRF -> ranked results"
            ),
            "hybrid_reranker": (
                "same Hybrid RRF candidates -> fixed CrossEncoder -> ranked results"
            ),
            "shared_candidate_rule": (
                "Within every query/pass, Hybrid and Hybrid + Reranker share "
                "the exact same RRF candidate objects and ranking."
            ),
        },
        "benchmark_environment": _runtime_environment(
            reranker_device=actual_reranker_device,
            reranker_model=str(config["reranker_model"]),
        ),
        "measurement_policy": {
            "clock": "time.perf_counter",
            "execution": "single-process, single-request, controlled sequential",
            "query_order": "fixed W7-T3 heldout query order in every pass",
            "cold_start_query_id": cold_query_id,
            "warmup_pass_count": 1,
            "warmup_query_count": len(resolved["evaluation_query_ids"]),
            "warmup_samples_in_aggregate": False,
            "measured_pass_count": measured_pass_count,
            "measured_query_count_per_pass": len(
                resolved["evaluation_query_ids"]
            ),
            "expected_sample_count_per_method": (
                measured_pass_count * len(resolved["evaluation_query_ids"])
            ),
            "timing_boundary": (
                "Starts immediately before each retrieval/rerank component call "
                "and stops immediately after it returns. Metric computation, "
                "result validation, JSON serialization, file I/O, and LLM work "
                "are outside the timed interval. Hybrid + Reranker total is the "
                "sum of its shared Hybrid stage and rerank stage."
            ),
            "percentile_method": (
                "linear interpolation at (n - 1) * percentile over sorted samples"
            ),
        },
        "cold_start": {
            "component_construction_ms": component_construction_ms,
            "lazy_model_was_unloaded_before_first_request": True,
            "fixed_stage_order": ["vector", "hybrid_rrf", "hybrid_reranker"],
            "vector_first_call_ms_including_embedding_initialization": (
                cold_start_samples[0]["total_retrieval_ms"]
            ),
            "hybrid_first_call_after_vector_initialization_ms": (
                cold_start_samples[1]["total_retrieval_ms"]
            ),
            "first_rerank_call_ms_including_lazy_model_load": (
                cold_start_samples[2]["rerank_ms"]
            ),
            "first_hybrid_reranker_sequential_total_ms": (
                cold_start_samples[2]["total_retrieval_ms"]
            ),
            "interpretation": (
                "The first rerank call includes lazy model loading, tokenizer/"
                "framework initialization, and first inference; it is not a pure "
                "model-load-only measurement and is excluded from warm aggregates. "
                "Cold stages ran in the recorded fixed order, so their values are "
                "initialization diagnostics and are not a fair cross-method cold "
                "latency comparison."
            ),
            "samples": cold_start_samples,
        },
        "warmup": {
            "excluded_from_aggregate": True,
            "samples": warmup_samples,
        },
        "per_sample_results": measured_samples,
        "aggregate_results": aggregate_results,
        "reranker_stage_latency": rerank_stage,
        "per_query_latency": aggregate_latency_by_query(measured_samples),
        "latency_deltas": latency_deltas,
        "quality_evidence": quality_evidence,
        "tradeoff_summary": {
            "primary_quality_k": quality_evidence["primary_k"],
            "hybrid_quality": quality_evidence["primary_metrics_by_method"][
                "hybrid_rrf"
            ],
            "hybrid_reranker_quality": quality_evidence[
                "primary_metrics_by_method"
            ]["hybrid_reranker"],
            "quality_delta": quality_evidence["quality_deltas"][
                "hybrid_to_hybrid_reranker"
            ],
            "hybrid_p50_ms": aggregate_results["hybrid_rrf"]["p50_ms"],
            "hybrid_reranker_p50_ms": aggregate_results["hybrid_reranker"][
                "p50_ms"
            ],
            "p50_delta_ms": latency_deltas["hybrid_to_hybrid_reranker"][
                "p50_ms"
            ]["absolute_delta_ms"],
            "p50_delta_percent": latency_deltas[
                "hybrid_to_hybrid_reranker"
            ]["p50_ms"]["relative_delta_percent"],
            "hybrid_p95_ms": aggregate_results["hybrid_rrf"]["p95_ms"],
            "hybrid_reranker_p95_ms": aggregate_results["hybrid_reranker"][
                "p95_ms"
            ],
            "p95_delta_ms": latency_deltas["hybrid_to_hybrid_reranker"][
                "p95_ms"
            ]["absolute_delta_ms"],
            "p95_delta_percent": latency_deltas[
                "hybrid_to_hybrid_reranker"
            ]["p95_ms"]["relative_delta_percent"],
            "rerank_stage_p50_ms": rerank_stage["p50_ms"],
            "measured_failure_count": failure_count,
        },
        "complexity_and_cost_boundary": {
            "local_reranker": True,
            "external_api_fee_introduced": False,
            "monetary_cost_measured": False,
            "local_compute_and_memory_required": True,
            "memory_usage_measured": False,
            "latency_sla_available": False,
        },
        "claim_boundary": [
            "These measurements characterize the current local environment, not a production SLA.",
            "Concurrency, throughput, saturation, and multi-worker behavior were not evaluated.",
            "LLM generation and end-to-end RAG answer latency were not measured.",
            "The five-query heldout quality set and local sequential workload limit generalization.",
            "No ranking parameter, model, batch behavior, corpus, or index was changed.",
        ],
    }
    _write_artifact(output_path, payload, overwrite=overwrite)
    return payload


def _validate_runtime_components(
    hybrid: Any,
    reranker: Any,
    *,
    config: Mapping[str, Any],
) -> None:
    bm25_retriever = hybrid.retrievers[1]
    if bm25_retriever.index.k1 != config["bm25_k1"]:
        raise ValueError("runtime BM25 k1 differs from frozen manifest")
    if bm25_retriever.index.b != config["bm25_b"]:
        raise ValueError("runtime BM25 b differs from frozen manifest")

    scorer = getattr(reranker, "_scorer", None)
    if scorer is None or not hasattr(scorer, "_model"):
        raise ValueError("default reranker does not expose its lazy model state")
    if scorer._model is not None:
        raise RuntimeError(
            "reranker model was already loaded; cold-start measurement is invalid"
        )


def _actual_reranker_device(reranker: Any) -> str:
    scorer = getattr(reranker, "_scorer", None)
    model = getattr(scorer, "_model", None)
    if model is None:
        raise RuntimeError("reranker model did not load during cold-start request")
    device = getattr(model, "device", None)
    if device is None:
        raise RuntimeError("loaded reranker does not expose its actual device")
    return str(device)


def _validate_measured_rankings(
    samples: Sequence[Mapping[str, Any]],
    *,
    quality_artifact: Mapping[str, Any],
) -> None:
    expected = {
        method: {
            row["query_id"]: row["retrieved_chunk_ids"]
            for row in quality_artifact["per_query_results"][method]
        }
        for method in METHOD_NAMES
    }
    for sample in samples:
        if sample["status"] == FAILED:
            continue
        method = sample["method"]
        query_id = sample["query_id"]
        if sample["result_chunk_ids"] != expected[method][query_id]:
            raise RuntimeError(
                f"{method} ranking drifted from W7-T3 for query {query_id}"
            )
        if method == "hybrid_reranker":
            expected_candidates = expected["hybrid_rrf"][query_id]
            if sample["pre_rerank_chunk_ids"] != expected_candidates:
                raise RuntimeError(
                    f"reranker candidates drifted from W7-T3 for query {query_id}"
                )


def _runtime_environment(
    *,
    reranker_device: str,
    reranker_model: str,
) -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    except ImportError:
        cuda_available = None
        cuda_device_count = None
    return {
        "os": platform.platform(),
        "python": sys.version,
        "cpu_identifier": platform.processor() or os.environ.get(
            "PROCESSOR_IDENTIFIER"
        ),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "reranker_device": reranker_device,
        "reranker_model": reranker_model,
        "libraries": {
            name: _package_version(name)
            for name in (
                "sentence-transformers",
                "torch",
                "transformers",
                "faiss-cpu",
                "numpy",
            )
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _repository_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "working_tree_dirty": None}


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{name} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_reference(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": _file_sha256(path),
    }


def _write_artifact(
    output_path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if output_path.resolve() in PROTECTED_INPUTS:
        raise ValueError("refusing to overwrite a frozen input artifact")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"artifact already exists: {output_path}; pass --overwrite explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the frozen W7 retrieval methods without tuning."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--quality-artifact", type=Path, default=DEFAULT_QUALITY_ARTIFACT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--measured-passes", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace only the W7-T4 latency artifact.",
    )
    args = parser.parse_args()

    payload = run_benchmark(
        manifest_path=_resolve(args.manifest),
        quality_artifact_path=_resolve(args.quality_artifact),
        output_path=_resolve(args.output),
        measured_pass_count=args.measured_passes,
        overwrite=args.overwrite,
    )
    print("W7-T4 frozen retrieval latency benchmark completed")
    print(f"- status: {payload['status']}")
    print(
        f"- samples per method: "
        f"{payload['measurement_policy']['expected_sample_count_per_method']}"
    )
    for method, aggregate in payload["aggregate_results"].items():
        print(
            f"- {method}: p50={aggregate['p50_ms']:.3f} ms, "
            f"p95={aggregate['p95_ms']:.3f} ms, n={aggregate['sample_count']}"
        )
    delta = payload["latency_deltas"]["hybrid_to_hybrid_reranker"]
    print(
        f"- Hybrid -> Hybrid + Reranker p50 delta: "
        f"{delta['p50_ms']['absolute_delta_ms']:.3f} ms "
        f"({delta['p50_ms']['relative_delta_percent']:.1f}%)"
    )
    print(f"- artifact: {_resolve(args.output)}")
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
