import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.evaluation.retrieval import (
    aggregate_retrieval_results,
    calculate_retrieval_metrics,
)
from app.evaluation.retrieval_comparison import (
    METHOD_NAMES,
    build_pairwise_comparisons,
    build_reranker_effect,
    validate_comparable_results,
)
from app.retrieval import HybridRetriever, VectorRetriever, get_default_reranker
from app.services.access_control import get_readable_document_ids
from app.services.knowledge_base import build_bm25_retriever
from scripts.evaluate_rag import load_dataset
from scripts.evaluate_retrieval import _file_sha256, validate_examples


DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "retrieval_evaluation_config.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evals" / "results" / "W7-T3-retrieval-evaluation.json"
)
PROTECTED_ARTIFACTS = {
    (PROJECT_ROOT / "evals" / "results" / "W6-T4-retrieval-evaluation.json").resolve(),
    (PROJECT_ROOT / "evals" / "results" / "W6-T5-failure-analysis.json").resolve(),
    (
        PROJECT_ROOT
        / "evals"
        / "results"
        / "W7-T2-retrieval-configuration.json"
    ).resolve(),
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _project_path(recorded_path: str) -> Path:
    path = PROJECT_ROOT / recorded_path
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {recorded_path}") from exc
    return path


def _validate_recorded_file(record: Mapping[str, Any], *, name: str) -> Path:
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name}.path must be a non-empty string")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{name}.sha256 must be a SHA-256 string")
    path = _project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"frozen file is missing: {path}")
    if _file_sha256(path).lower() != expected_hash.lower():
        raise ValueError(f"frozen file hash mismatch: {name}")
    return path


def validate_frozen_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate identities and W7-T2 values before any heldout query is run."""
    manifest = _load_json(manifest_path)
    if manifest.get("task") != "W7-T3":
        raise ValueError("frozen manifest task must be W7-T3")
    if manifest.get("created_before_heldout_results") is not True:
        raise ValueError("manifest must be frozen before heldout results")
    if tuple(manifest.get("methods", ())) != METHOD_NAMES:
        raise ValueError(f"methods must equal {list(METHOD_NAMES)}")

    source_artifact_path = _validate_recorded_file(
        manifest["source_w7_t2_artifact"], name="source_w7_t2_artifact"
    )
    dataset_path = _validate_recorded_file(manifest["dataset"], name="dataset")
    split_path = _validate_recorded_file(manifest["split"], name="split")
    for position, record in enumerate(manifest["corpus"]["documents"]):
        _validate_recorded_file(record, name=f"corpus.documents[{position}]")
    chunk_index_path = _validate_recorded_file(
        manifest["corpus"]["chunk_index"], name="corpus.chunk_index"
    )
    vector_index_path = _validate_recorded_file(
        manifest["corpus"]["vector_index"], name="corpus.vector_index"
    )
    _validate_recorded_file(manifest["corpus"]["faiss_index"], name="corpus.faiss_index")
    _validate_recorded_file(
        manifest["corpus"]["faiss_metadata"], name="corpus.faiss_metadata"
    )

    examples = load_dataset(dataset_path)
    validate_examples(examples)
    answerable_count = sum(
        1 for example in examples if bool(example.get("should_answer", True))
    )
    unanswerable_count = len(examples) - answerable_count
    dataset_record = manifest["dataset"]
    if len(examples) != dataset_record["total_query_count"]:
        raise ValueError("dataset total query count drifted")
    if answerable_count != dataset_record["answerable_query_count"]:
        raise ValueError("dataset answerable query count drifted")
    if unanswerable_count != dataset_record["unanswerable_query_count"]:
        raise ValueError("dataset unanswerable query count drifted")

    split = _load_json(split_path)
    evaluation_query_ids = manifest["split"]["evaluation_query_ids"]
    if evaluation_query_ids != split.get("heldout_query_ids"):
        raise ValueError("evaluation query IDs must equal the W7-T2 heldout IDs")
    if manifest["split"]["excluded_unanswerable_query_ids"] != split.get(
        "unanswerable_query_ids"
    ):
        raise ValueError("unanswerable exclusion differs from the W7-T2 split")
    all_query_ids = {f"q{position:03d}" for position in range(1, len(examples) + 1)}
    if any(query_id not in all_query_ids for query_id in evaluation_query_ids):
        raise ValueError("evaluation contains an unknown query ID")
    examples_by_id = {
        f"q{position:03d}": example
        for position, example in enumerate(examples, start=1)
    }
    if any(
        not bool(examples_by_id[query_id].get("should_answer", True))
        for query_id in evaluation_query_ids
    ):
        raise ValueError("primary heldout evaluation must contain answerable queries")

    w7_t2 = _load_json(source_artifact_path)
    config = manifest["resolved_configuration"]
    selected = w7_t2.get("selected_configuration_for_w7_t3", {})
    expected_selected = {
        "config_id": config["selected_w7_t2_config_id"],
        "per_source_candidate_depth": config["per_source_candidate_depth"],
        "rrf_k": config["rrf_k"],
        "rerank_candidate_count": config["reranker_candidate_count"],
        "final_top_k": config["operational_final_top_k"],
    }
    if selected != expected_selected:
        raise ValueError("resolved configuration differs from W7-T2 selection")
    if w7_t2["fixed_configuration"]["reranker_model"] != config["reranker_model"]:
        raise ValueError("reranker model differs from W7-T2")
    if w7_t2["fixed_configuration"]["vector_min_score"] != config["vector_min_score"]:
        raise ValueError("vector min_score differs from W7-T2")

    k_values = config["metric_k_values"]
    if (
        not isinstance(k_values, list)
        or not k_values
        or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in k_values)
        or k_values != sorted(set(k_values))
    ):
        raise ValueError("metric_k_values must be sorted unique positive integers")
    evaluation_depth = config["offline_evaluation_depth"]
    if max(k_values) > evaluation_depth:
        raise ValueError("offline evaluation depth must cover every metric K")
    if evaluation_depth != config["rrf_output_count"]:
        raise ValueError("offline depth must equal the frozen RRF output count")
    if config["rrf_output_count"] != config["reranker_candidate_count"]:
        raise ValueError("RRF output and reranker candidate count must match")
    if config["operational_final_top_k"] > config["reranker_candidate_count"]:
        raise ValueError("operational final_top_k exceeds reranker candidate count")

    chunks = _load_json_list(chunk_index_path, name="chunk index")
    vectors = _load_json_list(vector_index_path, name="vector index")
    expected_chunk_count = manifest["corpus"]["indexed_chunk_count"]
    if len(chunks) != expected_chunk_count or len(vectors) != expected_chunk_count:
        raise ValueError("frozen index chunk count drifted")
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    vector_ids = [chunk.get("chunk_id") for chunk in vectors]
    if len(set(chunk_ids)) != len(chunk_ids) or chunk_ids != vector_ids:
        raise ValueError("chunk/vector index identities differ")
    embedding_models = {vector.get("embedding_model") for vector in vectors}
    if embedding_models != {config["embedding_model"]}:
        raise ValueError("vector index embedding model differs from manifest")

    if settings.embedding_provider != "local_model":
        raise ValueError("frozen evaluation requires the local_model embedding provider")
    if settings.local_embedding_model != config["embedding_model"]:
        raise ValueError("runtime embedding model differs from frozen manifest")
    if settings.vector_store_backend != config["vector_store_backend"]:
        raise ValueError("runtime vector backend differs from frozen manifest")
    if settings.reranker_model_name != config["reranker_model"]:
        raise ValueError("runtime reranker model differs from frozen manifest")
    if settings.reranker_local_files_only != config["reranker_local_files_only"]:
        raise ValueError("runtime reranker loading policy differs from manifest")

    return {
        "manifest": manifest,
        "manifest_sha256": _file_sha256(manifest_path),
        "dataset_path": dataset_path,
        "split_path": split_path,
        "chunk_index_path": chunk_index_path,
        "vector_index_path": vector_index_path,
        "examples": examples,
        "examples_by_id": examples_by_id,
        "evaluation_query_ids": evaluation_query_ids,
        "config": config,
        "w7_t2": w7_t2,
        "split": split,
    }


def _load_json_list(path: Path, *, name: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON") from exc
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"{name} must contain a JSON list of objects")
    return payload


def build_method_components(
    *,
    chunk_index_path: Path,
    vector_index_path: Path,
    vector_min_score: float,
    allowed_document_ids: frozenset[str],
) -> tuple[VectorRetriever, HybridRetriever, Any]:
    standalone_vector = VectorRetriever(
        index_path=vector_index_path,
        min_score=vector_min_score,
        allowed_document_ids=allowed_document_ids,
    )
    hybrid_vector = VectorRetriever(
        index_path=vector_index_path,
        min_score=vector_min_score,
        allowed_document_ids=allowed_document_ids,
    )
    bm25 = build_bm25_retriever(
        chunk_index_path,
        allowed_document_ids=allowed_document_ids,
    )
    hybrid = HybridRetriever(
        [hybrid_vector, bm25],
        allowed_document_ids=allowed_document_ids,
    )
    return standalone_vector, hybrid, get_default_reranker()


def evaluate_query_methods(
    query_id: str,
    example: Mapping[str, Any],
    *,
    vector_retriever: Any,
    hybrid_retriever: Any,
    reranker: Any,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run all methods while sharing one exact Hybrid candidate list."""
    query = str(example["question"])
    evaluation_depth = int(config["offline_evaluation_depth"])
    k_values = list(config["metric_k_values"])

    vector_started = perf_counter()
    vector_results = vector_retriever.retrieve(query, evaluation_depth)
    vector_latency_ms = (perf_counter() - vector_started) * 1000.0

    hybrid_started = perf_counter()
    hybrid_results = hybrid_retriever.retrieve_fused(
        query,
        int(config["rrf_output_count"]),
        candidate_depth=int(config["per_source_candidate_depth"]),
        rrf_k=int(config["rrf_k"]),
    )
    hybrid_latency_ms = (perf_counter() - hybrid_started) * 1000.0

    rerank_started = perf_counter()
    reranked_results = reranker.rerank(
        query,
        hybrid_results,
        top_k=evaluation_depth,
    )
    rerank_latency_ms = (perf_counter() - rerank_started) * 1000.0

    _validate_ranked_results(vector_results, method="vector", max_count=evaluation_depth)
    _validate_ranked_results(
        hybrid_results, method="hybrid_rrf", max_count=evaluation_depth
    )
    _validate_ranked_results(
        reranked_results, method="hybrid_reranker", max_count=evaluation_depth
    )
    hybrid_ids = [result["chunk_id"] for result in hybrid_results]
    reranked_ids = [result["chunk_id"] for result in reranked_results]
    if len(hybrid_ids) != len(reranked_ids) or set(hybrid_ids) != set(reranked_ids):
        raise RuntimeError("reranker output is not a permutation of Hybrid candidates")

    return {
        "vector": _build_result_row(
            query_id,
            example,
            method="vector",
            ranked_results=vector_results,
            k_values=k_values,
            latency_ms=vector_latency_ms,
            stage_latency_ms={"vector": vector_latency_ms},
        ),
        "hybrid_rrf": _build_result_row(
            query_id,
            example,
            method="hybrid_rrf",
            ranked_results=hybrid_results,
            k_values=k_values,
            latency_ms=hybrid_latency_ms,
            stage_latency_ms={"hybrid_retrieval_and_fusion": hybrid_latency_ms},
        ),
        "hybrid_reranker": _build_result_row(
            query_id,
            example,
            method="hybrid_reranker",
            ranked_results=reranked_results,
            k_values=k_values,
            latency_ms=hybrid_latency_ms + rerank_latency_ms,
            stage_latency_ms={
                "shared_hybrid_retrieval_and_fusion": hybrid_latency_ms,
                "reranking": rerank_latency_ms,
            },
            pre_rerank_results=hybrid_results,
        ),
    }


def _validate_ranked_results(
    results: Sequence[Mapping[str, Any]],
    *,
    method: str,
    max_count: int,
) -> None:
    if len(results) > max_count:
        raise RuntimeError(f"{method} returned more than the evaluation depth")
    chunk_ids = []
    for rank, result in enumerate(results, start=1):
        for field in ("chunk_id", "document_id", "filename"):
            if not isinstance(result.get(field), str) or not result[field]:
                raise ValueError(f"{method} rank {rank} is missing {field}")
        chunk_ids.append(result["chunk_id"])
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"{method} returned duplicate chunk identities")


def _build_result_row(
    query_id: str,
    example: Mapping[str, Any],
    *,
    method: str,
    ranked_results: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
    latency_ms: float,
    stage_latency_ms: Mapping[str, float],
    pre_rerank_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    relevant_documents = list(example["expected_sources"])
    relevant_chunks_value = example.get("expected_citation_chunk_ids")
    relevant_chunks = (
        list(relevant_chunks_value)
        if isinstance(relevant_chunks_value, list) and relevant_chunks_value
        else None
    )
    retrieved_documents = [str(result["filename"]) for result in ranked_results]
    retrieved_chunks = [str(result["chunk_id"]) for result in ranked_results]
    row = {
        "query_id": query_id,
        "query": str(example["question"]),
        "method": method,
        "category": example.get("category", "uncategorized"),
        "difficulty": example.get("difficulty", "unknown"),
        "should_answer": True,
        "metrics_status": "evaluated",
        "relevance_label_type": "document_filename",
        "relevant_document_labels": relevant_documents,
        "relevant_chunk_ids": relevant_chunks,
        "retrieved_chunk_ids": retrieved_chunks,
        "retrieved_document_labels": retrieved_documents,
        "relevant_document_ranks": _all_ranks(
            ranked_results, relevant_documents, identity_field="filename"
        ),
        "relevant_chunk_ranks": (
            _all_ranks(ranked_results, relevant_chunks, identity_field="chunk_id")
            if relevant_chunks
            else None
        ),
        "ranked_results": _ranked_diagnostics(ranked_results),
        "metrics_by_k": {
            str(k): {
                "k": k,
                **calculate_retrieval_metrics(
                    retrieved_documents,
                    relevant_documents,
                    top_k=k,
                ),
            }
            for k in k_values
        },
        "strict_chunk_metrics_by_k": (
            {
                str(k): {
                    "k": k,
                    **calculate_retrieval_metrics(
                        retrieved_chunks,
                        relevant_chunks,
                        top_k=k,
                    ),
                }
                for k in k_values
            }
            if relevant_chunks
            else None
        ),
        "latency_ms": latency_ms,
        "stage_latency_sanity_ms": dict(stage_latency_ms),
    }
    if pre_rerank_results is not None:
        row["pre_rerank_chunk_ids"] = [
            result["chunk_id"] for result in pre_rerank_results
        ]
        row["pre_rerank_results"] = _ranked_diagnostics(pre_rerank_results)
    return row


def _all_ranks(
    results: Sequence[Mapping[str, Any]],
    relevant_ids: Sequence[str],
    *,
    identity_field: str,
) -> dict[str, list[int]]:
    ranks = {relevant_id: [] for relevant_id in relevant_ids}
    for rank, result in enumerate(results, start=1):
        identity = result.get(identity_field)
        if identity in ranks:
            ranks[str(identity)].append(rank)
    return ranks


def _ranked_diagnostics(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "chunk_id",
        "document_id",
        "filename",
        "position",
        "chunk_index",
        "page_number",
        "source",
        "retrieval_mode",
        "score",
        "fused_score",
        "rerank_score",
        "matched_sources",
        "source_ranks",
        "source_scores",
        "embedding_model",
        "query_embedding_model",
        "vector_store_backend",
        "created_at",
        "metadata",
    )
    return [
        {"rank": rank, **{field: result.get(field) for field in fields}}
        for rank, result in enumerate(results, start=1)
    ]


def _aggregate_strict_subset(
    results_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    summary = {}
    query_ids: list[str] | None = None
    for method, results in results_by_method.items():
        strict_rows = [
            {
                **result,
                "metrics_by_k": result["strict_chunk_metrics_by_k"],
            }
            for result in results
            if result["strict_chunk_metrics_by_k"] is not None
        ]
        method_ids = [row["query_id"] for row in strict_rows]
        if query_ids is None:
            query_ids = method_ids
        elif method_ids != query_ids:
            raise ValueError("strict chunk query IDs differ across methods")
        summary[method] = aggregate_retrieval_results(strict_rows, k_values=k_values)
    return {
        "label_type": "chunk_id",
        "query_count": len(query_ids or []),
        "query_ids": query_ids or [],
        "summary": summary,
        "limitation": (
            "Diagnostic only: the heldout subset has strict chunk labels only for q027."
        ),
    }


def _multi_relevant_analysis(
    results_by_method: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    indexed = {
        method: {row["query_id"]: row for row in rows}
        for method, rows in results_by_method.items()
    }
    reference = results_by_method["vector"]
    analyses = []
    for row in reference:
        if len(row["relevant_document_labels"]) <= 1:
            continue
        query_id = row["query_id"]
        analyses.append(
            {
                "query_id": query_id,
                "relevant_document_labels": row["relevant_document_labels"],
                "relevant_chunk_ids": row["relevant_chunk_ids"],
                "methods": {
                    method: {
                        "retrieved_chunk_ids": indexed[method][query_id][
                            "retrieved_chunk_ids"
                        ],
                        "relevant_document_ranks": indexed[method][query_id][
                            "relevant_document_ranks"
                        ],
                        "relevant_chunk_ranks": indexed[method][query_id][
                            "relevant_chunk_ranks"
                        ],
                        "metrics_by_k": indexed[method][query_id]["metrics_by_k"],
                        "strict_chunk_metrics_by_k": indexed[method][query_id][
                            "strict_chunk_metrics_by_k"
                        ],
                    }
                    for method in METHOD_NAMES
                },
            }
        )
    return analyses


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


def write_artifact(
    output_path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    if output_path.resolve() in PROTECTED_ARTIFACTS:
        raise ValueError("refusing to overwrite a protected W6/W7-T2 artifact")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"artifact already exists: {output_path}; pass --overwrite explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_evaluation(
    *,
    manifest_path: Path,
    output_path: Path,
    user_id: UUID,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved = validate_frozen_manifest(manifest_path)
    config = resolved["config"]
    allowed_document_ids = get_readable_document_ids(user_id)
    vector, hybrid, reranker = build_method_components(
        chunk_index_path=resolved["chunk_index_path"],
        vector_index_path=resolved["vector_index_path"],
        vector_min_score=float(config["vector_min_score"]),
        allowed_document_ids=allowed_document_ids,
    )
    bm25_retriever = hybrid.retrievers[1]
    if bm25_retriever.index.k1 != config["bm25_k1"]:
        raise ValueError("runtime BM25 k1 differs from frozen manifest")
    if bm25_retriever.index.b != config["bm25_b"]:
        raise ValueError("runtime BM25 b differs from frozen manifest")

    # Warm all fixed components with one tuning query. Heldout outputs are not
    # inspected before the formal loop, and timing remains only a sanity signal.
    warmup_id = resolved["split"]["tuning_query_ids"][0]
    warmup_example = resolved["examples_by_id"][warmup_id]
    evaluate_query_methods(
        warmup_id,
        warmup_example,
        vector_retriever=vector,
        hybrid_retriever=hybrid,
        reranker=reranker,
        config=config,
    )

    results_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHOD_NAMES
    }
    for query_id in resolved["evaluation_query_ids"]:
        rows = evaluate_query_methods(
            query_id,
            resolved["examples_by_id"][query_id],
            vector_retriever=vector,
            hybrid_retriever=hybrid,
            reranker=reranker,
            config=config,
        )
        for method in METHOD_NAMES:
            results_by_method[method].append(rows[method])

    k_values = list(config["metric_k_values"])
    validated_query_ids = validate_comparable_results(
        results_by_method, k_values=k_values
    )
    aggregate_results = {
        method: aggregate_retrieval_results(results, k_values=k_values)
        for method, results in results_by_method.items()
    }
    pairwise = build_pairwise_comparisons(results_by_method, k_values=k_values)
    reranker_effect = build_reranker_effect(
        results_by_method["hybrid_rrf"],
        results_by_method["hybrid_reranker"],
        k_values=k_values,
    )

    manifest = resolved["manifest"]
    payload = {
        "artifact_version": 1,
        "task": "W7-T3",
        "experiment_id": manifest["experiment_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Heldout retrieval/ranking quality only: Vector vs Hybrid RRF vs "
            "Hybrid RRF + fixed Reranker. No LLM generation and no tuning."
        ),
        "repository_state": _repository_state(),
        "frozen_manifest": {
            "path": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": resolved["manifest_sha256"],
            "created_before_heldout_results": True,
        },
        "source_artifacts": {
            "w7_t2": manifest["source_w7_t2_artifact"],
            "split": {
                "path": manifest["split"]["path"],
                "sha256": manifest["split"]["sha256"],
            },
        },
        "dataset_identity": {
            **manifest["dataset"],
            "evaluation_query_ids": validated_query_ids,
            "primary_evaluation_query_count": len(validated_query_ids),
            "primary_evaluation_scope": "heldout_answerable_only",
        },
        "split_identity": {
            **manifest["split"],
            "tuning_results_included": False,
            "unanswerable_results_included": False,
        },
        "corpus_identity": manifest["corpus"],
        "resolved_configuration": config,
        "authorization_user_id": str(user_id),
        "method_configurations": {
            "vector": {
                "flow": "VectorRetriever -> ranked results",
                "evaluation_depth": config["offline_evaluation_depth"],
                "embedding_model": config["embedding_model"],
                "vector_store_backend": config["vector_store_backend"],
                "similarity": config["vector_similarity"],
                "min_score": config["vector_min_score"],
            },
            "hybrid_rrf": {
                "flow": "VectorRetriever + BM25Retriever -> RRF -> ranked results",
                "per_source_candidate_depth": config[
                    "per_source_candidate_depth"
                ],
                "rrf_k": config["rrf_k"],
                "rrf_output_count": config["rrf_output_count"],
                "reranker_used": False,
            },
            "hybrid_reranker": {
                "flow": "same Hybrid RRF candidates -> fixed Cross-Encoder -> ranked results",
                "per_source_candidate_depth": config[
                    "per_source_candidate_depth"
                ],
                "rrf_k": config["rrf_k"],
                "reranker_candidate_count": config["reranker_candidate_count"],
                "reranker_model": config["reranker_model"],
                "offline_output_depth": config["offline_evaluation_depth"],
                "operational_final_top_k": config["operational_final_top_k"],
            },
        },
        "metric_definition": {
            "label_type": "document_filename",
            "k_values": k_values,
            "hit_rate": "at least one relevant document occurs in Top-K",
            "recall": "unique relevant documents in Top-K / all relevant documents",
            "mrr": "reciprocal rank of the first relevant document within Top-K",
            "comparison_rule": manifest["comparison_rule"],
        },
        "aggregate_results": aggregate_results,
        "strict_chunk_labeled_subset": _aggregate_strict_subset(
            results_by_method, k_values=k_values
        ),
        "pairwise_comparisons": pairwise,
        "reranker_effect": reranker_effect,
        "multi_relevant_queries": _multi_relevant_analysis(results_by_method),
        "per_query_results": results_by_method,
        "unanswerable_handling": {
            "query_ids": manifest["split"]["excluded_unanswerable_query_ids"],
            "included_in_standard_relevance_aggregate": False,
            "policy": manifest["unanswerable_policy"],
        },
        "timing_policy": (
            "Warmup used one tuning query. Timing is a local sanity signal only, "
            "is not used for quality conclusions, and does not replace W7-T4."
        ),
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sentence_transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "faiss_cpu": _package_version("faiss-cpu"),
        },
        "claim_boundary": [
            "The run evaluates five heldout answerable queries only.",
            "Primary metrics are document-level; q027 is the only strict chunk diagnostic.",
            "The evaluation does not measure LLM answer quality or latency trade-offs.",
            "The frozen configuration was not changed after heldout results.",
        ],
    }
    write_artifact(output_path, payload, overwrite=overwrite)
    return payload


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen W7-T3 heldout retrieval comparison."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL user whose readable documents define the heldout corpus.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace only the W7-T3 artifact after a technical fix.",
    )
    args = parser.parse_args()

    payload = run_evaluation(
        manifest_path=_resolve(args.manifest),
        output_path=_resolve(args.output),
        user_id=args.user_id,
        overwrite=args.overwrite,
    )
    print("W7-T3 heldout retrieval evaluation completed")
    print(f"- query IDs: {payload['dataset_identity']['evaluation_query_ids']}")
    print(f"- metric K values: {payload['metric_definition']['k_values']}")
    for method, aggregate in payload["aggregate_results"].items():
        print(f"- {method}: {json.dumps(aggregate['metrics_by_k'])}")
    print(f"- artifact: {_resolve(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
