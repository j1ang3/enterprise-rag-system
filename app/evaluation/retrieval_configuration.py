"""Controlled configuration primitives for the W7-T2 retrieval experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence


QUALITY_TOLERANCE = 1e-12
MIN_MULTI_RELEVANT_FINAL_TOP_K = 2


class FusedRetriever(Protocol):
    def retrieve_fused(
        self,
        query: str,
        top_k: int,
        *,
        candidate_depth: int | None = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]: ...


class CandidateReranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    """All ranking knobs that must stay explicit in a controlled run."""

    per_source_candidate_depth: int
    rrf_k: int
    rerank_candidate_count: int
    final_top_k: int

    def __post_init__(self) -> None:
        for field_name in (
            "per_source_candidate_depth",
            "rrf_k",
            "rerank_candidate_count",
            "final_top_k",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.final_top_k > self.rerank_candidate_count:
            raise ValueError(
                "final_top_k must not exceed rerank_candidate_count because "
                "the final cutoff is applied after reranking"
            )

    @property
    def config_id(self) -> str:
        """Return a readable ID determined only by the serialized configuration."""
        return (
            f"ps{self.per_source_candidate_depth:02d}"
            f"-r{self.rrf_k:03d}"
            f"-n{self.rerank_candidate_count:02d}"
            f"-k{self.final_top_k:02d}"
        )

    def to_dict(self) -> dict[str, int | str]:
        return {"config_id": self.config_id, **asdict(self)}


@dataclass(frozen=True)
class PipelineRun:
    candidates_before_rerank: list[dict[str, Any]]
    results_after_rerank: list[dict[str, Any]]
    retrieval_latency_ms: float
    reranking_latency_ms: float


def build_candidate_count_matrix(
    baseline: RetrievalExperimentConfig,
    candidate_counts: Sequence[int],
) -> list[RetrievalExperimentConfig]:
    configs = [
        replace(baseline, rerank_candidate_count=count)
        for count in candidate_counts
    ]
    validate_one_variable_matrix(configs, variable="rerank_candidate_count")
    return configs


def build_final_top_k_matrix(
    fixed_candidate_config: RetrievalExperimentConfig,
    final_top_k_values: Sequence[int],
) -> list[RetrievalExperimentConfig]:
    configs = [
        replace(fixed_candidate_config, final_top_k=top_k)
        for top_k in final_top_k_values
    ]
    validate_one_variable_matrix(configs, variable="final_top_k")
    return configs


def validate_one_variable_matrix(
    configs: Sequence[RetrievalExperimentConfig],
    *,
    variable: str,
) -> None:
    fields = {
        "per_source_candidate_depth",
        "rrf_k",
        "rerank_candidate_count",
        "final_top_k",
    }
    if variable not in fields:
        raise ValueError(f"unsupported experiment variable: {variable}")
    if len(configs) < 2:
        raise ValueError("an experiment matrix must contain at least two configurations")
    if len({config.config_id for config in configs}) != len(configs):
        raise ValueError("an experiment matrix must not contain duplicate configurations")

    frozen_fields = fields - {variable}
    first = configs[0]
    for config in configs[1:]:
        for field_name in frozen_fields:
            if getattr(config, field_name) != getattr(first, field_name):
                raise ValueError(
                    f"only {variable} may vary; {field_name} changed as well"
                )


def run_configured_pipeline(
    query: str,
    config: RetrievalExperimentConfig,
    *,
    hybrid_retriever: FusedRetriever,
    reranker: CandidateReranker,
) -> PipelineRun:
    """Run fusion then reranking, with each cutoff applied at its own stage."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    retrieval_started = perf_counter()
    candidates = hybrid_retriever.retrieve_fused(
        normalized_query,
        config.rerank_candidate_count,
        candidate_depth=config.per_source_candidate_depth,
        rrf_k=config.rrf_k,
    )
    retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000.0
    if len(candidates) > config.rerank_candidate_count:
        raise RuntimeError("fused retriever returned more candidates than requested")

    reranking_started = perf_counter()
    reranked = reranker.rerank(
        normalized_query,
        candidates,
        top_k=config.final_top_k,
    )
    reranking_latency_ms = (perf_counter() - reranking_started) * 1000.0
    if len(reranked) > config.final_top_k:
        raise RuntimeError("reranker returned more results than final_top_k")

    candidate_ids = {candidate.get("chunk_id") for candidate in candidates}
    if any(result.get("chunk_id") not in candidate_ids for result in reranked):
        raise RuntimeError("reranker returned a result outside the fused candidate set")

    return PipelineRun(
        candidates_before_rerank=candidates,
        results_after_rerank=reranked,
        retrieval_latency_ms=retrieval_latency_ms,
        reranking_latency_ms=reranking_latency_ms,
    )


def select_candidate_configuration(
    configs: Sequence[RetrievalExperimentConfig],
    aggregates: Mapping[str, Mapping[str, float]],
    *,
    baseline_config_id: str,
) -> RetrievalExperimentConfig:
    """Apply the predeclared no-regression rule, then prefer the smaller pool."""
    baseline = _baseline_metrics(configs, aggregates, baseline_config_id)
    eligible = [
        config
        for config in configs
        if _not_lower(
            aggregates[config.config_id]["candidate_recall"],
            baseline["candidate_recall"],
        )
        and _not_lower(
            aggregates[config.config_id]["final_recall"],
            baseline["final_recall"],
        )
        and _not_lower(
            aggregates[config.config_id]["final_mrr"], baseline["final_mrr"]
        )
    ]
    return max(
        eligible,
        key=lambda config: (
            aggregates[config.config_id]["candidate_recall"],
            aggregates[config.config_id]["final_recall"],
            aggregates[config.config_id]["final_mrr"],
            -config.rerank_candidate_count,
        ),
    )


def select_final_top_k_configuration(
    configs: Sequence[RetrievalExperimentConfig],
    aggregates: Mapping[str, Mapping[str, float]],
    *,
    baseline_config_id: str,
) -> RetrievalExperimentConfig:
    """Keep multi-relevant capacity and quality, then prefer the smaller final K."""
    baseline = _baseline_metrics(configs, aggregates, baseline_config_id)
    eligible = [
        config
        for config in configs
        if config.final_top_k >= MIN_MULTI_RELEVANT_FINAL_TOP_K
        and _not_lower(
            aggregates[config.config_id]["final_recall"],
            baseline["final_recall"],
        )
        and _not_lower(
            aggregates[config.config_id]["final_mrr"], baseline["final_mrr"]
        )
    ]
    return max(
        eligible,
        key=lambda config: (
            aggregates[config.config_id]["final_recall"],
            aggregates[config.config_id]["final_mrr"],
            aggregates[config.config_id]["final_hit_rate"],
            -config.final_top_k,
        ),
    )


def _baseline_metrics(
    configs: Sequence[RetrievalExperimentConfig],
    aggregates: Mapping[str, Mapping[str, float]],
    baseline_config_id: str,
) -> Mapping[str, float]:
    config_ids = {config.config_id for config in configs}
    if baseline_config_id not in config_ids:
        raise ValueError("baseline configuration is not present in the matrix")
    missing = config_ids - set(aggregates)
    if missing:
        raise ValueError(f"aggregate metrics are missing for: {sorted(missing)}")
    return aggregates[baseline_config_id]


def _not_lower(value: float, baseline: float) -> bool:
    return float(value) + QUALITY_TOLERANCE >= float(baseline)
