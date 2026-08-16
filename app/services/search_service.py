from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import AbstractSet, Callable, Dict, List

from app.retrieval.base import RetrievalResult
from app.retrieval.fusion import DEFAULT_RRF_K
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import SemanticReranker, get_default_reranker
from app.retrieval.vector import VectorRetriever
from app.services.knowledge_base import build_bm25_retriever


@dataclass(frozen=True)
class RerankedHybridConfig:
    """Resolved candidate and cutoff budgets for Hybrid + RRF + Reranker."""

    per_source_candidate_depth: int = 5
    rrf_k: int = DEFAULT_RRF_K
    rerank_candidate_count: int = 3
    final_top_k: int = 2

    def __post_init__(self) -> None:
        values = (
            self.per_source_candidate_depth,
            self.rrf_k,
            self.rerank_candidate_count,
            self.final_top_k,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("reranked hybrid configuration values must be positive integers")
        if self.final_top_k > self.rerank_candidate_count:
            raise ValueError("final_top_k cannot exceed rerank_candidate_count")

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RerankedHybridSearchResult:
    candidates_before_rerank: List[RetrievalResult]
    results_after_rerank: List[RetrievalResult]
    configuration: RerankedHybridConfig
    retrieval_ms: float | None = None
    rerank_ms: float | None = None


def build_hybrid_retriever(
    *,
    chunk_index_path: Path | None = None,
    vector_index_path: Path | None = None,
    allowed_document_ids: AbstractSet[str],
) -> HybridRetriever:
    """Compose existing semantic and lexical retrievers for candidate collection."""
    return HybridRetriever(
        (
            VectorRetriever(
                index_path=vector_index_path,
                min_score=0.0,
                allowed_document_ids=allowed_document_ids,
            ),
            build_bm25_retriever(
                chunk_index_path,
                allowed_document_ids=allowed_document_ids,
            ),
        ),
        allowed_document_ids=allowed_document_ids,
    )


def search_hybrid_chunks(
    query: str,
    top_k: int = 5,
    *,
    chunk_index_path: Path | None = None,
    vector_index_path: Path | None = None,
    allowed_document_ids: AbstractSet[str],
) -> List[RetrievalResult]:
    """Return un-fused candidates from every configured retrieval source."""
    retriever = build_hybrid_retriever(
        chunk_index_path=chunk_index_path,
        vector_index_path=vector_index_path,
        allowed_document_ids=allowed_document_ids,
    )
    return retriever.retrieve(query, top_k)


def search_fused_hybrid_chunks(
    query: str,
    top_k: int = 5,
    *,
    candidate_depth: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    chunk_index_path: Path | None = None,
    vector_index_path: Path | None = None,
    allowed_document_ids: AbstractSet[str],
) -> List[RetrievalResult]:
    """Return de-duplicated, RRF-ranked results from hybrid candidates."""
    retriever = build_hybrid_retriever(
        chunk_index_path=chunk_index_path,
        vector_index_path=vector_index_path,
        allowed_document_ids=allowed_document_ids,
    )
    return retriever.retrieve_fused(
        query,
        top_k=top_k,
        candidate_depth=candidate_depth,
        rrf_k=rrf_k,
    )


def run_reranked_hybrid_search(
    query: str,
    *,
    configuration: RerankedHybridConfig,
    chunk_index_path: Path | None = None,
    vector_index_path: Path | None = None,
    reranker: SemanticReranker | None = None,
    clock: Callable[[], float] = perf_counter,
    allowed_document_ids: AbstractSet[str],
) -> RerankedHybridSearchResult:
    """Run the W7 Hybrid/RRF candidate flow and semantic reranker as one component."""
    retriever = build_hybrid_retriever(
        chunk_index_path=chunk_index_path,
        vector_index_path=vector_index_path,
        allowed_document_ids=allowed_document_ids,
    )
    retrieval_started = clock()
    candidates = retriever.retrieve_fused(
        query,
        top_k=configuration.rerank_candidate_count,
        candidate_depth=configuration.per_source_candidate_depth,
        rrf_k=configuration.rrf_k,
    )
    candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate.get("document_id"), str)
        and bool(candidate["document_id"])
        and candidate["document_id"] in allowed_document_ids
    ]
    retrieval_ms = round((clock() - retrieval_started) * 1000, 3)
    rerank_started = clock()
    ranked = (reranker or get_default_reranker()).rerank(
        query,
        candidates,
        top_k=configuration.final_top_k,
    )
    rerank_ms = round((clock() - rerank_started) * 1000, 3)
    final_results: List[RetrievalResult] = [
        {**result, "retrieval_mode": "hybrid_rerank"}
        for result in ranked
    ]
    return RerankedHybridSearchResult(
        candidates_before_rerank=candidates,
        results_after_rerank=final_results,
        configuration=configuration,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
    )


def search_reranked_hybrid_chunks(
    query: str,
    *,
    configuration: RerankedHybridConfig,
    chunk_index_path: Path | None = None,
    vector_index_path: Path | None = None,
    reranker: SemanticReranker | None = None,
    allowed_document_ids: AbstractSet[str],
) -> List[RetrievalResult]:
    return run_reranked_hybrid_search(
        query,
        configuration=configuration,
        chunk_index_path=chunk_index_path,
        vector_index_path=vector_index_path,
        reranker=reranker,
        allowed_document_ids=allowed_document_ids,
    ).results_after_rerank
