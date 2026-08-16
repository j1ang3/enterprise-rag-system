"""Reusable comparison and rank diagnostics for retrieval evaluation artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from app.evaluation.retrieval import classify_rank_movement


METHOD_NAMES = ("vector", "hybrid_rrf", "hybrid_reranker")
PAIR_SPECS = (
    ("vector_vs_hybrid", "vector", "hybrid_rrf"),
    ("vector_vs_hybrid_reranker", "vector", "hybrid_reranker"),
    ("hybrid_vs_hybrid_reranker", "hybrid_rrf", "hybrid_reranker"),
)


def validate_comparable_results(
    results_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Sequence[int],
    expected_methods: Sequence[str] = METHOD_NAMES,
) -> list[str]:
    """Require identical queries, labels, and metric cutoffs across methods."""
    if tuple(results_by_method) != tuple(expected_methods):
        raise ValueError(f"methods must equal {list(expected_methods)} in that order")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    expected_query_ids: list[str] | None = None
    reference_rows: dict[str, Mapping[str, Any]] = {}
    for method in expected_methods:
        results = results_by_method[method]
        if not results:
            raise ValueError(f"{method} results must not be empty")
        query_ids = [str(result.get("query_id", "")) for result in results]
        if any(not query_id for query_id in query_ids) or len(set(query_ids)) != len(
            query_ids
        ):
            raise ValueError(f"{method} query IDs must be non-empty and unique")
        if expected_query_ids is None:
            expected_query_ids = query_ids
            reference_rows = {result["query_id"]: result for result in results}
        elif query_ids != expected_query_ids:
            raise ValueError("all methods must use the same ordered query IDs")

        for result in results:
            if result.get("method") != method:
                raise ValueError(f"result method does not match container: {method}")
            if result.get("metrics_status") != "evaluated":
                raise ValueError("primary comparison requires answerable labeled queries")
            metric_keys = set(result.get("metrics_by_k", {}))
            if metric_keys != {str(k) for k in k_values}:
                raise ValueError("all methods must use the same metric K values")
            relevant = result.get("relevant_document_labels")
            if not isinstance(relevant, list) or not relevant:
                raise ValueError("evaluated queries require document relevance labels")

            if method != expected_methods[0]:
                reference = reference_rows[result["query_id"]]
                if result.get("query") != reference.get("query"):
                    raise ValueError("query text differs across methods")
                if relevant != reference.get("relevant_document_labels"):
                    raise ValueError("relevance labels differ across methods")
    return expected_query_ids or []


def build_pairwise_comparisons(
    results_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Group challenger outcomes by lexicographic (Recall@K, MRR@K)."""
    query_ids = validate_comparable_results(results_by_method, k_values=k_values)
    indexed = {
        method: {result["query_id"]: result for result in results}
        for method, results in results_by_method.items()
    }
    comparisons: dict[str, Any] = {}
    for comparison_id, baseline_method, challenger_method in PAIR_SPECS:
        by_k = {}
        for k in k_values:
            groups = {
                "challenger_better": [],
                "challenger_worse": [],
                "equal": [],
            }
            for query_id in query_ids:
                baseline = indexed[baseline_method][query_id]
                challenger = indexed[challenger_method][query_id]
                baseline_quality = _quality(baseline, k)
                challenger_quality = _quality(challenger, k)
                detail = {
                    "query_id": query_id,
                    "query": baseline["query"],
                    "category": baseline.get("category", "uncategorized"),
                    "baseline_quality": {
                        "recall": baseline_quality[0],
                        "mrr": baseline_quality[1],
                    },
                    "challenger_quality": {
                        "recall": challenger_quality[0],
                        "mrr": challenger_quality[1],
                    },
                    "baseline_first_relevant_rank": baseline["metrics_by_k"][str(k)][
                        "first_relevant_rank"
                    ],
                    "challenger_first_relevant_rank": challenger["metrics_by_k"][str(k)][
                        "first_relevant_rank"
                    ],
                    "baseline_chunk_ids": baseline["retrieved_chunk_ids"],
                    "challenger_chunk_ids": challenger["retrieved_chunk_ids"],
                }
                if challenger_quality > baseline_quality:
                    groups["challenger_better"].append(detail)
                elif challenger_quality < baseline_quality:
                    groups["challenger_worse"].append(detail)
                else:
                    groups["equal"].append(detail)
            by_k[str(k)] = {
                **groups,
                "empty_groups": [name for name, values in groups.items() if not values],
            }
        comparisons[comparison_id] = {
            "baseline_method": baseline_method,
            "challenger_method": challenger_method,
            "by_k": by_k,
        }
    return {
        "comparison_definition": (
            "At each K compare (Recall@K, MRR@K) lexicographically; exact ties "
            "are equal."
        ),
        "comparisons": comparisons,
    }


def build_reranker_effect(
    hybrid_results: Sequence[Mapping[str, Any]],
    reranked_results: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Explain how the fixed reranker changed the exact Hybrid candidate lists."""
    if [row.get("query_id") for row in hybrid_results] != [
        row.get("query_id") for row in reranked_results
    ]:
        raise ValueError("Hybrid and reranker results must use identical query IDs")

    items = []
    for hybrid, reranked in zip(hybrid_results, reranked_results, strict=True):
        pre_ranked = hybrid["ranked_results"]
        post_ranked = reranked["ranked_results"]
        pre_ids = [result["chunk_id"] for result in pre_ranked]
        recorded_pre_ids = reranked.get("pre_rerank_chunk_ids")
        if pre_ids != recorded_pre_ids:
            raise ValueError("reranker input differs from the Hybrid candidate ranking")
        post_ids = [result["chunk_id"] for result in post_ranked]
        if set(post_ids) != set(pre_ids) or len(post_ids) != len(pre_ids):
            raise ValueError("reranker output must be a permutation of Hybrid candidates")

        relevant_documents = hybrid["relevant_document_labels"]
        relevant_chunks = hybrid.get("relevant_chunk_ids") or []
        document_movements = _relevant_movements(
            pre_ranked,
            post_ranked,
            relevant_documents,
            identity_field="filename",
            k_values=k_values,
        )
        strict_chunk_movements = _relevant_movements(
            pre_ranked,
            post_ranked,
            relevant_chunks,
            identity_field="chunk_id",
            k_values=k_values,
        )
        quality_equal_by_k = {
            str(k): _quality(hybrid, k) == _quality(reranked, k) for k in k_values
        }
        ordering_changed = pre_ids != post_ids
        items.append(
            {
                "query_id": hybrid["query_id"],
                "query": hybrid["query"],
                "category": hybrid.get("category", "uncategorized"),
                "pre_rerank_chunk_ids": pre_ids,
                "post_rerank_chunk_ids": post_ids,
                "ordering_changed": ordering_changed,
                "quality_equal_by_k": quality_equal_by_k,
                "rank_changed_but_metric_unchanged_by_k": {
                    str(k): ordering_changed and quality_equal_by_k[str(k)]
                    for k in k_values
                },
                "candidate_recall_failure_document_level": any(
                    movement["pre_rerank_first_rank"] is None
                    for movement in document_movements
                ),
                "document_level_movements": document_movements,
                "strict_chunk_level_movements": strict_chunk_movements,
            }
        )

    return {
        "rank_delta_definition": (
            "pre_rerank_rank - post_rerank_rank; positive values mean promotion"
        ),
        "items": items,
        "summary": _summarize_reranker_effect(items, k_values=k_values),
    }


def _quality(result: Mapping[str, Any], k: int) -> tuple[float, float]:
    metrics = result["metrics_by_k"][str(k)]
    return float(metrics["recall"]), float(metrics["reciprocal_rank"])


def _rank_positions(
    ranked_results: Sequence[Mapping[str, Any]],
    relevant_ids: Sequence[str],
    *,
    identity_field: str,
) -> dict[str, list[int]]:
    positions = {identity: [] for identity in relevant_ids}
    for rank, result in enumerate(ranked_results, start=1):
        identity = result.get(identity_field)
        if identity in positions:
            positions[str(identity)].append(rank)
    return positions


def _relevant_movements(
    pre_ranked: Sequence[Mapping[str, Any]],
    post_ranked: Sequence[Mapping[str, Any]],
    relevant_ids: Sequence[str],
    *,
    identity_field: str,
    k_values: Sequence[int],
) -> list[dict[str, Any]]:
    pre_positions = _rank_positions(
        pre_ranked, relevant_ids, identity_field=identity_field
    )
    post_positions = _rank_positions(
        post_ranked, relevant_ids, identity_field=identity_field
    )
    movements = []
    for relevant_id in relevant_ids:
        pre_ranks = pre_positions[relevant_id]
        post_ranks = post_positions[relevant_id]
        pre_first = pre_ranks[0] if pre_ranks else None
        post_first = post_ranks[0] if post_ranks else None
        movements.append(
            {
                "relevant_id": relevant_id,
                "pre_rerank_ranks": pre_ranks,
                "post_rerank_ranks": post_ranks,
                "pre_rerank_first_rank": pre_first,
                "post_rerank_first_rank": post_first,
                "rank_delta": (
                    pre_first - post_first
                    if pre_first is not None and post_first is not None
                    else None
                ),
                "candidate_retrieved": pre_first is not None,
                "movement_by_k": {
                    str(k): classify_rank_movement(pre_first, post_first, k=k)
                    for k in k_values
                },
            }
        )
    return movements


def _summarize_reranker_effect(
    items: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    direction_counts: Counter[str] = Counter()
    strict_direction_counts: Counter[str] = Counter()
    for item in items:
        for movement in item["document_level_movements"]:
            direction_counts[_direction_from_delta(movement)] += 1
        for movement in item["strict_chunk_level_movements"]:
            strict_direction_counts[_direction_from_delta(movement)] += 1

    return {
        "query_count": len(items),
        "ordering_changed_query_ids": [
            item["query_id"] for item in items if item["ordering_changed"]
        ],
        "no_ranking_change_query_ids": [
            item["query_id"] for item in items if not item["ordering_changed"]
        ],
        "candidate_recall_failure_query_ids": [
            item["query_id"]
            for item in items
            if item["candidate_recall_failure_document_level"]
        ],
        "rank_changed_but_metric_unchanged_query_ids_by_k": {
            str(k): [
                item["query_id"]
                for item in items
                if item["rank_changed_but_metric_unchanged_by_k"][str(k)]
            ]
            for k in k_values
        },
        "document_relevant_item_direction_counts": dict(direction_counts),
        "strict_chunk_relevant_item_direction_counts": dict(strict_direction_counts),
        "cutoff_effect_counts_by_k": {
            str(k): dict(
                Counter(
                    movement["movement_by_k"][str(k)]["cutoff_effect"]
                    for item in items
                    for movement in item["document_level_movements"]
                )
            )
            for k in k_values
        },
    }


def _direction_from_delta(movement: Mapping[str, Any]) -> str:
    delta = movement["rank_delta"]
    if movement["pre_rerank_first_rank"] is None:
        return "candidate_not_retrieved"
    if movement["post_rerank_first_rank"] is None:
        return "no_longer_observed"
    if delta > 0:
        return "promoted"
    if delta < 0:
        return "demoted"
    return "unchanged"
