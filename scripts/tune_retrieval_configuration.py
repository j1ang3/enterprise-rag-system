import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.evaluation.retrieval import calculate_retrieval_metrics
from app.evaluation.retrieval_configuration import (
    RetrievalExperimentConfig,
    build_candidate_count_matrix,
    build_final_top_k_matrix,
    run_configured_pipeline,
    select_candidate_configuration,
    select_final_top_k_configuration,
)
from app.retrieval import HybridRetriever, VectorRetriever, get_default_reranker
from app.services.access_control import get_readable_document_ids
from app.services.knowledge_base import build_bm25_retriever
from scripts.evaluate_rag import bootstrap_documents, load_dataset
from scripts.evaluate_retrieval import validate_examples


DEFAULT_DATASET = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "evals" / "retrieval_configuration_split.json"
DEFAULT_BOOTSTRAP_DOCS = (
    PROJECT_ROOT / "eval_docs" / "hr_policy.md",
    PROJECT_ROOT / "eval_docs" / "expense_policy.md",
    PROJECT_ROOT / "eval_docs" / "security_policy.md",
    PROJECT_ROOT / "eval_docs" / "product_faq.md",
)
DEFAULT_CHUNK_INDEX = PROJECT_ROOT / "storage" / "eval" / "week07_tuning_chunks.json"
DEFAULT_VECTOR_INDEX = PROJECT_ROOT / "storage" / "eval" / "week07_tuning_vectors.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "evals" / "results" / "W7-T2-retrieval-configuration.json"
)
PROTECTED_W6_ARTIFACT = (
    PROJECT_ROOT / "evals" / "results" / "W6-T4-retrieval-evaluation.json"
)

BASELINE_CONFIG = RetrievalExperimentConfig(
    per_source_candidate_depth=5,
    rrf_k=60,
    rerank_candidate_count=5,
    final_top_k=3,
)
CANDIDATE_COUNT_VALUES = (5, 3, 8)
FINAL_TOP_K_VALUES = (3, 1, 2)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_MIN_SCORE = 0.0
BM25_K1 = 1.5
BM25_B = 0.75

SELECTION_RULE = {
    "candidate_count": (
        "Require Candidate Recall@N, final Recall@K, and final MRR@K to be "
        "no lower than the baseline; maximize those quality metrics in that "
        "order, then prefer the smaller candidate count on a complete tie."
    ),
    "final_top_k": (
        "Require final_top_k >= 2 for the known multi-relevant use case and "
        "require final Recall@K and MRR@K to be no lower than the K=3 "
        "baseline; maximize quality, then prefer the smaller K on a tie."
    ),
    "excluded_factors": (
        "Do not select by one query or latency. Heldout and unanswerable "
        "queries are not evaluated during W7-T2 tuning."
    ),
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_split_manifest(
    manifest: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    dataset_sha256: str,
) -> dict[str, list[str]]:
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("split manifest dataset_sha256 does not match the dataset")
    if manifest.get("created_before_experiment_results") is not True:
        raise ValueError("split manifest must declare pre-result creation")

    all_ids = [f"q{position:03d}" for position in range(1, len(examples) + 1)]
    groups: dict[str, list[str]] = {}
    for field_name in (
        "tuning_query_ids",
        "heldout_query_ids",
        "unanswerable_query_ids",
    ):
        values = manifest.get(field_name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field_name} must be a non-empty list")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{field_name} must contain non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{field_name} contains duplicate query IDs")
        groups[field_name] = list(values)

    flattened = [query_id for values in groups.values() for query_id in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("split groups must be disjoint")
    if set(flattened) != set(all_ids):
        raise ValueError("split groups must cover every dataset query exactly once")
    if "q027" not in groups["heldout_query_ids"]:
        raise ValueError("q027 must remain held out during W7-T2")

    examples_by_id = dict(zip(all_ids, examples, strict=True))
    for query_id in groups["tuning_query_ids"] + groups["heldout_query_ids"]:
        if not bool(examples_by_id[query_id].get("should_answer", True)):
            raise ValueError(f"answerable split contains unanswerable query: {query_id}")
    for query_id in groups["unanswerable_query_ids"]:
        if bool(examples_by_id[query_id].get("should_answer", True)):
            raise ValueError(f"unanswerable split contains answerable query: {query_id}")
    return groups


def select_tuning_examples(
    examples: Sequence[Mapping[str, Any]],
    tuning_query_ids: Sequence[str],
) -> list[tuple[str, Mapping[str, Any]]]:
    examples_by_id = {
        f"q{position:03d}": example
        for position, example in enumerate(examples, start=1)
    }
    selected = [(query_id, examples_by_id[query_id]) for query_id in tuning_query_ids]
    if [query_id for query_id, _ in selected] != list(tuning_query_ids):
        raise RuntimeError("tuning query order changed unexpectedly")
    return selected


def _ranked_diagnostics(
    results: Sequence[Mapping[str, Any]],
    *,
    include_rerank_score: bool,
) -> list[dict[str, Any]]:
    diagnostics = []
    for rank, result in enumerate(results, start=1):
        item = {
            "rank": rank,
            "chunk_id": result.get("chunk_id"),
            "filename": result.get("filename"),
            "fused_score": result.get("fused_score"),
            "matched_sources": result.get("matched_sources"),
            "source_ranks": result.get("source_ranks"),
            "source_scores": result.get("source_scores"),
        }
        if include_rerank_score:
            item["rerank_score"] = result.get("rerank_score")
        diagnostics.append(item)
    return diagnostics


def _all_relevant_ranks(
    results: Sequence[Mapping[str, Any]],
    relevant_ids: Sequence[str],
    *,
    identity_field: str,
) -> dict[str, list[int]]:
    # A relevant document can contribute multiple chunks, so retaining only one
    # rank would conceal either its first hit or its lower-ranked duplicates.
    ranks = {relevant_id: [] for relevant_id in relevant_ids}
    for rank, result in enumerate(results, start=1):
        identity = result.get(identity_field)
        if identity in ranks:
            ranks[str(identity)].append(rank)
    return ranks


def evaluate_configuration(
    config: RetrievalExperimentConfig,
    tuning_examples: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    hybrid_retriever: HybridRetriever,
    reranker: Any,
) -> list[dict[str, Any]]:
    per_query = []
    for query_id, example in tuning_examples:
        query = str(example["question"])
        run = run_configured_pipeline(
            query,
            config,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        )
        relevant_documents = list(example["expected_sources"])
        relevant_chunks_value = example.get("expected_citation_chunk_ids")
        relevant_chunks = (
            list(relevant_chunks_value)
            if isinstance(relevant_chunks_value, list) and relevant_chunks_value
            else None
        )

        before_documents = [
            str(candidate["filename"]) for candidate in run.candidates_before_rerank
        ]
        after_documents = [
            str(result["filename"]) for result in run.results_after_rerank
        ]
        candidate_metrics = calculate_retrieval_metrics(
            before_documents,
            relevant_documents,
            top_k=config.rerank_candidate_count,
        )
        final_metrics = calculate_retrieval_metrics(
            after_documents,
            relevant_documents,
            top_k=config.final_top_k,
        )

        strict_chunk_metrics = None
        if relevant_chunks:
            strict_chunk_metrics = {
                "candidate": calculate_retrieval_metrics(
                    [
                        str(candidate["chunk_id"])
                        for candidate in run.candidates_before_rerank
                    ],
                    relevant_chunks,
                    top_k=config.rerank_candidate_count,
                ),
                "final": calculate_retrieval_metrics(
                    [str(result["chunk_id"]) for result in run.results_after_rerank],
                    relevant_chunks,
                    top_k=config.final_top_k,
                ),
            }

        per_query.append(
            {
                "configuration_id": config.config_id,
                "query_id": query_id,
                "query": query,
                "category": example.get("category", "uncategorized"),
                "difficulty": example.get("difficulty", "unknown"),
                "ground_truth": {
                    "relevant_document_labels": relevant_documents,
                    "relevant_chunk_ids": relevant_chunks,
                    "primary_label_type": "document_filename",
                },
                "configured_rerank_candidate_count": config.rerank_candidate_count,
                "actual_reranker_input_count": len(run.candidates_before_rerank),
                "configured_final_top_k": config.final_top_k,
                "actual_final_result_count": len(run.results_after_rerank),
                "before_rerank": _ranked_diagnostics(
                    run.candidates_before_rerank,
                    include_rerank_score=False,
                ),
                "after_rerank": _ranked_diagnostics(
                    run.results_after_rerank,
                    include_rerank_score=True,
                ),
                "relevant_document_ranks": {
                    "before": _all_relevant_ranks(
                        run.candidates_before_rerank,
                        relevant_documents,
                        identity_field="filename",
                    ),
                    "after": _all_relevant_ranks(
                        run.results_after_rerank,
                        relevant_documents,
                        identity_field="filename",
                    ),
                },
                "candidate_metrics": candidate_metrics,
                "final_metrics": final_metrics,
                "strict_chunk_metrics": strict_chunk_metrics,
                "latency_sanity_ms": {
                    "retrieval": run.retrieval_latency_ms,
                    "reranking": run.reranking_latency_ms,
                    "total": run.retrieval_latency_ms + run.reranking_latency_ms,
                },
            }
        )
    return per_query


def aggregate_configuration(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("configuration results must not be empty")
    strict_results = [result for result in results if result["strict_chunk_metrics"]]

    def average(path: Sequence[str], values: Sequence[Mapping[str, Any]]) -> float:
        resolved = []
        for value in values:
            current: Any = value
            for key in path:
                current = current[key]
            resolved.append(float(current))
        return mean(resolved) if resolved else 0.0

    return {
        "query_count": len(results),
        "primary_label_type": "document_filename",
        "candidate_recall": average(("candidate_metrics", "recall"), results),
        "final_hit_rate": mean(
            1.0 if result["final_metrics"]["hit"] else 0.0 for result in results
        ),
        "final_recall": average(("final_metrics", "recall"), results),
        "final_mrr": average(("final_metrics", "reciprocal_rank"), results),
        "strict_chunk_diagnostic": {
            "query_count": len(strict_results),
            "query_ids": [result["query_id"] for result in strict_results],
            "candidate_recall": average(
                ("strict_chunk_metrics", "candidate", "recall"), strict_results
            ),
            "final_hit_rate": mean(
                1.0 if result["strict_chunk_metrics"]["final"]["hit"] else 0.0
                for result in strict_results
            )
            if strict_results
            else 0.0,
            "final_recall": average(
                ("strict_chunk_metrics", "final", "recall"), strict_results
            ),
            "final_mrr": average(
                ("strict_chunk_metrics", "final", "reciprocal_rank"),
                strict_results,
            ),
            "limitation": (
                "Diagnostic only: W7-T2 tuning contains two queries with strict "
                "chunk labels; q027 remains held out."
            ),
        },
        "latency_sanity_ms": {
            "retrieval_mean": average(("latency_sanity_ms", "retrieval"), results),
            "reranking_mean": average(("latency_sanity_ms", "reranking"), results),
            "total_mean": average(("latency_sanity_ms", "total"), results),
            "selection_usage": "not_used; formal latency trade-off is W7-T4",
        },
        "actual_reranker_input_count": {
            "min": min(int(result["actual_reranker_input_count"]) for result in results),
            "max": max(int(result["actual_reranker_input_count"]) for result in results),
            "mean": mean(
                int(result["actual_reranker_input_count"]) for result in results
            ),
        },
    }


def _selection_metrics(aggregate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "candidate_recall": float(aggregate["candidate_recall"]),
        "final_hit_rate": float(aggregate["final_hit_rate"]),
        "final_recall": float(aggregate["final_recall"]),
        "final_mrr": float(aggregate["final_mrr"]),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def write_artifact(
    output_path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    if output_path.resolve() == PROTECTED_W6_ARTIFACT.resolve():
        raise ValueError("refusing to overwrite the W6-T4 artifact")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"artifact already exists: {output_path}; pass --overwrite explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_experiment(
    *,
    dataset_path: Path,
    split_manifest_path: Path,
    bootstrap_docs: Sequence[Path],
    chunk_index_path: Path,
    vector_index_path: Path,
    output_path: Path,
    user_id: UUID,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_path.resolve() == PROTECTED_W6_ARTIFACT.resolve():
        raise ValueError("W7-T2 output must not target the W6-T4 artifact")

    examples = load_dataset(dataset_path)
    validate_examples(examples)
    dataset_sha256 = _file_sha256(dataset_path)
    manifest = _load_json(split_manifest_path)
    groups = validate_split_manifest(
        manifest,
        examples,
        dataset_sha256=dataset_sha256,
    )
    tuning_examples = select_tuning_examples(
        examples,
        groups["tuning_query_ids"],
    )

    indexed_documents = bootstrap_documents(
        bootstrap_docs,
        chunk_index_path,
        vector_index_path,
    )
    allowed_document_ids = get_readable_document_ids(user_id)
    vector = VectorRetriever(
        index_path=vector_index_path,
        min_score=VECTOR_MIN_SCORE,
        allowed_document_ids=allowed_document_ids,
    )
    bm25 = build_bm25_retriever(
        chunk_index_path,
        allowed_document_ids=allowed_document_ids,
    )
    hybrid = HybridRetriever(
        [vector, bm25],
        allowed_document_ids=allowed_document_ids,
    )
    reranker = get_default_reranker()

    # Warm the fixed model with a tuning query so model-load time is not mixed into
    # the latency sanity samples. No heldout query is retrieved or reranked here.
    warmup_query = str(tuning_examples[0][1]["question"])
    warmup_candidates = hybrid.retrieve_fused(
        warmup_query,
        1,
        candidate_depth=BASELINE_CONFIG.per_source_candidate_depth,
        rrf_k=BASELINE_CONFIG.rrf_k,
    )
    reranker.rerank(warmup_query, warmup_candidates, top_k=1)

    candidate_configs = build_candidate_count_matrix(
        BASELINE_CONFIG,
        CANDIDATE_COUNT_VALUES,
    )
    results_by_config: dict[str, list[dict[str, Any]]] = {}
    aggregates_by_config: dict[str, dict[str, Any]] = {}
    for config in candidate_configs:
        results = evaluate_configuration(
            config,
            tuning_examples,
            hybrid_retriever=hybrid,
            reranker=reranker,
        )
        results_by_config[config.config_id] = results
        aggregates_by_config[config.config_id] = aggregate_configuration(results)

    candidate_selection = select_candidate_configuration(
        candidate_configs,
        {
            config.config_id: _selection_metrics(
                aggregates_by_config[config.config_id]
            )
            for config in candidate_configs
        },
        baseline_config_id=BASELINE_CONFIG.config_id,
    )

    fixed_candidate_config = replace(
        BASELINE_CONFIG,
        rerank_candidate_count=candidate_selection.rerank_candidate_count,
    )
    final_configs = build_final_top_k_matrix(
        fixed_candidate_config,
        FINAL_TOP_K_VALUES,
    )
    for config in final_configs:
        if config.config_id in results_by_config:
            continue
        results = evaluate_configuration(
            config,
            tuning_examples,
            hybrid_retriever=hybrid,
            reranker=reranker,
        )
        results_by_config[config.config_id] = results
        aggregates_by_config[config.config_id] = aggregate_configuration(results)

    final_selection = select_final_top_k_configuration(
        final_configs,
        {
            config.config_id: _selection_metrics(
                aggregates_by_config[config.config_id]
            )
            for config in final_configs
        },
        baseline_config_id=fixed_candidate_config.config_id,
    )

    all_configs = {config.config_id: config for config in candidate_configs + final_configs}
    selected_config = replace(
        final_selection,
        rerank_candidate_count=candidate_selection.rerank_candidate_count,
    )
    payload = {
        "task": "W7-T2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Tuning-set-only Hybrid+RRF+Reranker configuration experiment; "
            "not production integration and not W7-T3 heldout evaluation."
        ),
        "dataset_identity": {
            "path": _relative(dataset_path),
            "sha256": dataset_sha256,
            "total_query_count": len(examples),
            "primary_ground_truth_granularity": "document_filename",
            "strict_chunk_labeled_query_count": sum(
                1 for example in examples if example.get("expected_citation_chunk_ids")
            ),
        },
        "split_identity": {
            "path": _relative(split_manifest_path),
            "sha256": _file_sha256(split_manifest_path),
            "random_seed": manifest.get("random_seed"),
            **groups,
            "executed_query_ids": groups["tuning_query_ids"],
            "heldout_results_present": False,
        },
        "corpus_identity": {
            "documents": [
                {"path": _relative(path), "sha256": _file_sha256(path)}
                for path in bootstrap_docs
            ],
            "indexed_documents": indexed_documents,
            "indexed_chunk_count": sum(
                int(document["chunk_count"]) for document in indexed_documents
            ),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunk_index_path": _relative(chunk_index_path),
            "chunk_index_sha256": _file_sha256(chunk_index_path),
            "vector_index_path": _relative(vector_index_path),
            "vector_index_sha256": _file_sha256(vector_index_path),
        },
        "fixed_configuration": {
            "reranker_model": settings.reranker_model_name,
            "reranker_local_files_only": settings.reranker_local_files_only,
            "vector_min_score": VECTOR_MIN_SCORE,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "per_source_candidate_depth": BASELINE_CONFIG.per_source_candidate_depth,
            "rrf_k": BASELINE_CONFIG.rrf_k,
            "optional_extra_retrieval_parameter_tuned": None,
        },
        "baseline_configuration": BASELINE_CONFIG.to_dict(),
        "selection_rule_declared_before_results": SELECTION_RULE,
        "experiment_groups": {
            "candidate_count": {
                "experiment_id": "W7-T2-candidate-count",
                "changed_variable": "rerank_candidate_count",
                "config_ids_in_execution_order": [
                    config.config_id for config in candidate_configs
                ],
                "selected_config_id": candidate_selection.config_id,
            },
            "final_top_k": {
                "experiment_id": "W7-T2-final-top-k",
                "changed_variable": "final_top_k",
                "fixed_candidate_config_id": fixed_candidate_config.config_id,
                "config_ids_in_execution_order": [
                    config.config_id for config in final_configs
                ],
                "selected_config_id": final_selection.config_id,
                "baseline_result_reused": True,
            },
        },
        "configurations": {
            config_id: config.to_dict() for config_id, config in all_configs.items()
        },
        "aggregate_results": aggregates_by_config,
        "per_query_results": results_by_config,
        "selected_configuration_for_w7_t3": selected_config.to_dict(),
        "selection_rationale": {
            "candidate_count": (
                f"Applied the predeclared no-regression rule and selected "
                f"N={candidate_selection.rerank_candidate_count}."
            ),
            "final_top_k": (
                f"Applied the predeclared multi-relevant/no-regression rule and "
                f"selected K={final_selection.final_top_k}."
            ),
            "claim_boundary": (
                "This is a tuning-set recommendation for W7-T3. Heldout quality "
                "has not been measured and superiority is not established."
            ),
        },
        "latency_policy": (
            "Recorded only as a sanity signal; not used for selection and not a "
            "substitute for the W7-T4 latency trade-off experiment."
        ),
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sentence_transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "known_limitations": [
            "Primary labels are document-level and can hide chunk-ranking errors.",
            "Only q025 and q026 provide strict chunk diagnostics in tuning; q027 is held out.",
            "The corpus has only four documents and twelve chunks, so metrics may saturate.",
            "No production Search/Chat/RAG default is changed by this experiment.",
        ],
    }
    write_artifact(output_path, payload, overwrite=overwrite)
    return payload


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled W7-T2 retrieval configuration experiment."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--bootstrap-docs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_BOOTSTRAP_DOCS),
    )
    parser.add_argument("--chunk-index-path", type=Path, default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--vector-index-path", type=Path, default=DEFAULT_VECTOR_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL user whose readable documents define the experiment corpus.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing W7-T2 output artifact; W6 remains protected.",
    )
    args = parser.parse_args()

    payload = run_experiment(
        dataset_path=_resolve(args.dataset),
        split_manifest_path=_resolve(args.split_manifest),
        bootstrap_docs=[_resolve(path) for path in args.bootstrap_docs],
        chunk_index_path=_resolve(args.chunk_index_path),
        vector_index_path=_resolve(args.vector_index_path),
        output_path=_resolve(args.output),
        user_id=args.user_id,
        overwrite=args.overwrite,
    )
    selected = payload["selected_configuration_for_w7_t3"]
    print("W7-T2 retrieval configuration experiment completed")
    print(f"- tuning queries: {len(payload['split_identity']['executed_query_ids'])}")
    print(
        "- selected for W7-T3: "
        f"candidate_count={selected['rerank_candidate_count']}, "
        f"final_top_k={selected['final_top_k']}"
    )
    print(f"- artifact: {_resolve(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
