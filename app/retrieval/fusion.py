from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from app.retrieval.base import RetrievalResult


DEFAULT_RRF_K = 60
RankedSourceResults = Sequence[Tuple[str, Sequence[Mapping[str, Any]]]]


def _merge_missing_metadata(
    canonical_result: RetrievalResult,
    additional_result: Mapping[str, Any],
) -> None:
    canonical_metadata = canonical_result.get("metadata")
    additional_metadata = additional_result.get("metadata")
    if not isinstance(canonical_metadata, dict) or not isinstance(additional_metadata, Mapping):
        return

    for key, value in additional_metadata.items():
        canonical_metadata.setdefault(key, deepcopy(value))


def reciprocal_rank_fusion(
    ranked_results: RankedSourceResults,
    *,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[RetrievalResult]:
    """Fuse source-ranked chunks with RRF, using chunk_id as identity."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    fused_by_chunk_id: Dict[str, Dict[str, Any]] = {}
    seen_sources = set()
    first_seen_order = 0

    for source, source_results in ranked_results:
        if not source:
            raise ValueError("retrieval source must not be empty")
        if source in seen_sources:
            raise ValueError(f"duplicate retrieval source: {source}")
        seen_sources.add(source)

        seen_chunk_ids = set()
        for source_rank, result in enumerate(source_results, start=1):
            chunk_id = result.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("each retrieval result must contain a non-empty chunk_id")

            # A source should contribute only its best rank for a chunk, even if
            # a faulty backend returns that chunk more than once.
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            entry = fused_by_chunk_id.get(chunk_id)
            if entry is None:
                canonical_result = deepcopy(dict(result))
                entry = {
                    "result": canonical_result,
                    "fused_score": 0.0,
                    "matched_sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                    "best_source_rank": source_rank,
                    "first_seen_order": first_seen_order,
                }
                fused_by_chunk_id[chunk_id] = entry
                first_seen_order += 1
            else:
                canonical_result = entry["result"]
                _merge_missing_metadata(canonical_result, result)
                entry["best_source_rank"] = min(entry["best_source_rank"], source_rank)

            entry["fused_score"] += 1.0 / (rrf_k + source_rank)
            entry["matched_sources"].append(source)
            entry["source_ranks"][source] = source_rank

            raw_score = result.get("score")
            entry["source_scores"][source] = (
                float(raw_score)
                if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool)
                else None
            )

    fused_entries = sorted(
        fused_by_chunk_id.values(),
        key=lambda entry: (
            -entry["fused_score"],
            entry["best_source_rank"],
            entry["first_seen_order"],
            entry["result"]["chunk_id"],
        ),
    )

    fused_results = []
    for entry in fused_entries[:top_k]:
        fused_score = entry["fused_score"]
        fused_results.append(
            {
                **entry["result"],
                "source": "rrf",
                "retrieval_mode": "rrf",
                "score": fused_score,
                "fused_score": fused_score,
                "matched_sources": entry["matched_sources"],
                "source_ranks": entry["source_ranks"],
                "source_scores": entry["source_scores"],
            }
        )
    return fused_results
