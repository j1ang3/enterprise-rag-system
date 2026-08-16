from statistics import mean, median
from typing import Any, Dict, Iterable, Mapping, Sequence


def _validate_identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-empty strings")
    return value


def calculate_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    top_k: int,
) -> Dict[str, Any]:
    """Calculate binary hit, recall, and reciprocal rank at one cutoff.

    Identities are deliberately generic. W6-T4 uses document filenames because
    that is the complete human relevance judgement available in the fixed Week 5
    dataset; the same function can use chunk IDs once full chunk judgements exist.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    validated_retrieved = [
        _validate_identity(identity, field_name="retrieved_ids")
        for identity in retrieved_ids
    ]
    relevant_set = {
        _validate_identity(identity, field_name="relevant_ids")
        for identity in relevant_ids
    }
    if not relevant_set:
        raise ValueError("relevant_ids must contain at least one relevance label")

    retrieved_at_k = validated_retrieved[:top_k]
    matched_relevant_ids = sorted(set(retrieved_at_k) & relevant_set)
    first_relevant_rank = next(
        (
            rank
            for rank, identity in enumerate(retrieved_at_k, start=1)
            if identity in relevant_set
        ),
        None,
    )

    return {
        "hit": first_relevant_rank is not None,
        "recall": len(matched_relevant_ids) / len(relevant_set),
        "reciprocal_rank": (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
        "matched_relevant_ids": matched_relevant_ids,
    }


def aggregate_retrieval_results(
    results: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """Macro-average per-query metrics and summarize measured latency."""
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    evaluated = [result for result in results if result.get("metrics_status") == "evaluated"]
    latencies = [float(result["latency_ms"]) for result in results]

    metrics_by_k: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        key = str(k)
        if evaluated:
            metrics_by_k[key] = {
                "hit_rate": mean(
                    1.0 if result["metrics_by_k"][key]["hit"] else 0.0
                    for result in evaluated
                ),
                "recall": mean(
                    float(result["metrics_by_k"][key]["recall"])
                    for result in evaluated
                ),
                "mrr": mean(
                    float(result["metrics_by_k"][key]["reciprocal_rank"])
                    for result in evaluated
                ),
            }
        else:
            metrics_by_k[key] = {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}

    return {
        "query_count": len(results),
        "evaluated_query_count": len(evaluated),
        "not_applicable_query_count": len(results) - len(evaluated),
        "metrics_by_k": metrics_by_k,
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "sample_count": len(latencies),
        },
    }


def classify_rank_movement(
    source_rank: int | None,
    target_rank: int | None,
    *,
    k: int,
) -> Dict[str, str]:
    """Describe direction and cutoff effects between two ranked outputs."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if source_rank is None and target_rank is None:
        return {
            "direction": "not_observed_in_either",
            "cutoff_effect": "insufficient_evidence_outside_observed_depth",
        }
    if source_rank is None:
        return {
            "direction": "newly_observed",
            "cutoff_effect": (
                "crossed_into_top_k" if target_rank <= k else "observed_outside_top_k"
            ),
        }
    if target_rank is None:
        return {
            "direction": "no_longer_observed",
            "cutoff_effect": (
                "crossed_out_of_top_k"
                if source_rank <= k
                else "insufficient_evidence_outside_observed_depth"
            ),
        }
    if target_rank < source_rank:
        if source_rank > k >= target_rank:
            cutoff_effect = "crossed_into_top_k"
        elif source_rank > k and target_rank > k:
            cutoff_effect = "moved_up_but_did_not_cross_cutoff"
        else:
            cutoff_effect = "moved_up_within_top_k"
        return {"direction": "improved", "cutoff_effect": cutoff_effect}
    if target_rank > source_rank:
        if source_rank <= k < target_rank:
            cutoff_effect = "crossed_out_of_top_k"
        elif source_rank > k and target_rank > k:
            cutoff_effect = "moved_down_outside_top_k"
        else:
            cutoff_effect = "moved_down_within_top_k"
        return {"direction": "declined", "cutoff_effect": cutoff_effect}
    return {
        "direction": "unchanged",
        "cutoff_effect": (
            "unchanged_inside_top_k" if source_rank <= k else "unchanged_outside_top_k"
        ),
    }
