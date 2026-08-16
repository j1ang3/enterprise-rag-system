from copy import deepcopy
from typing import AbstractSet, Any, Dict, List, Mapping, Sequence

from app.retrieval.base import RetrievalResult, Retriever
from app.retrieval.fusion import DEFAULT_RRF_K, RankedSourceResults, reciprocal_rank_fusion


SOURCE_METADATA_FIELDS = (
    "document_id",
    "filename",
    "position",
    "chunk_index",
    "page_number",
    "token_count",
    "created_at",
)
REQUIRED_RESULT_FIELDS = ("chunk_id", "document_id", "filename")


def normalize_retrieval_result(
    result: Mapping[str, Any],
    source: str,
) -> RetrievalResult:
    """Return one source result in the shared hybrid-candidate shape."""
    missing_fields = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing_fields:
        raise ValueError(
            f"retrieval result is missing required fields: {', '.join(missing_fields)}"
        )

    content = result.get("content", result.get("text"))
    if not isinstance(content, str):
        raise ValueError("retrieval result must contain string content or text")

    raw_metadata = result.get("metadata")
    metadata: Dict[str, Any] = (
        deepcopy(dict(raw_metadata))
        if isinstance(raw_metadata, Mapping)
        else {}
    )
    for field in SOURCE_METADATA_FIELDS:
        if field in result:
            metadata.setdefault(field, deepcopy(result[field]))

    return {
        **deepcopy(dict(result)),
        "content": content,
        "text": content,
        "metadata": metadata,
        "source": source,
    }


class HybridRetriever:
    """Coordinate independent retrievers and optional rank fusion."""

    source = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        allowed_document_ids: AbstractSet[str],
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.retrievers = tuple(retrievers)
        self.allowed_document_ids = frozenset(allowed_document_ids)

    def _retrieve_ranked_sources(
        self,
        query: str,
        candidate_depth: int,
    ) -> RankedSourceResults:
        ranked_sources = []
        for retriever in self.retrievers:
            source_results = retriever.retrieve(query, candidate_depth)
            authorized_results = [
                result
                for result in source_results
                if isinstance(result.get("document_id"), str)
                and bool(result["document_id"])
                and result["document_id"] in self.allowed_document_ids
            ]
            ranked_sources.append(
                (
                    retriever.source,
                    [
                        normalize_retrieval_result(result, retriever.source)
                        for result in authorized_results
                    ],
                )
            )
        return ranked_sources

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip():
            return []

        ranked_sources = self._retrieve_ranked_sources(query, top_k)
        return [result for _, source_results in ranked_sources for result in source_results]

    def retrieve_fused(
        self,
        query: str,
        top_k: int = 5,
        *,
        candidate_depth: int | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[RetrievalResult]:
        """Retrieve per-source candidates, then return RRF-ranked final results."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        effective_candidate_depth = candidate_depth if candidate_depth is not None else top_k
        if effective_candidate_depth <= 0:
            raise ValueError("candidate_depth must be greater than 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if not query.strip():
            return []

        ranked_sources = self._retrieve_ranked_sources(query, effective_candidate_depth)
        return reciprocal_rank_fusion(ranked_sources, top_k=top_k, rrf_k=rrf_k)
