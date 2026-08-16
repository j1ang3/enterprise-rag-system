from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.core.config import settings
from app.db.session import DatabaseConfigurationError
from app.schemas.search import (
    HybridSearchRequest,
    HybridSearchResponse,
    RankFusionSearchRequest,
    RankFusionSearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.schemas.common import ErrorResponse
from app.services.embeddings import EmbeddingClientError
from app.services.access_control import (
    AccessControlUnavailableError,
    get_readable_document_ids,
)
from app.services.search_service import search_fused_hybrid_chunks, search_hybrid_chunks
from app.services.vector_store import search_vector_chunks
from app.services.user_registry import UserIdentity
from app.utils.response import success_response


router = APIRouter(prefix="/search", tags=["search"])


def _readable_document_ids(user_id: UUID) -> frozenset[str]:
    try:
        return get_readable_document_ids(user_id)
    except (DatabaseConfigurationError, AccessControlUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document authorization is temporarily unavailable.",
        ) from exc


@router.post(
    "",
    response_model=SemanticSearchResponse,
    summary="Run permission-aware semantic search",
    description=(
        "Retrieve vector-similar chunks only from documents the authenticated user "
        "owns or can read through an explicit ACL grant. No LLM is called."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Bearer credentials are missing or invalid.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Authorization or semantic embedding is unavailable.",
        },
    },
)
def semantic_search(
    request: SemanticSearchRequest,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    """Retrieve semantically similar chunks without calling an LLM."""
    min_score = request.min_score
    if min_score is None:
        min_score = settings.vector_min_score

    allowed_document_ids = _readable_document_ids(current_user.user_id)
    try:
        results = search_vector_chunks(
            request.query,
            top_k=request.top_k,
            min_score=min_score,
            allowed_document_ids=allowed_document_ids,
        )
    except EmbeddingClientError as exc:
        raise HTTPException(
            status_code=503,
            detail="Semantic search is temporarily unavailable.",
        ) from exc

    return success_response(
        data={
            "query": request.query,
            "top_k": request.top_k,
            "min_score": min_score,
            "result_count": len(results),
            "results": results,
        },
        message="semantic search completed",
    )


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    summary="Collect permission-aware hybrid candidates",
    description=(
        "Collect vector and BM25 candidates from readable documents without rank "
        "fusion or LLM generation."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Bearer credentials are missing or invalid.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Authorization or hybrid retrieval is unavailable.",
        },
    },
)
def hybrid_search(
    request: HybridSearchRequest,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    """Collect vector and BM25 candidates without calling an LLM or fusing ranks."""
    allowed_document_ids = _readable_document_ids(current_user.user_id)
    try:
        results = search_hybrid_chunks(
            request.query,
            top_k=request.top_k,
            allowed_document_ids=allowed_document_ids,
        )
    except EmbeddingClientError as exc:
        raise HTTPException(
            status_code=503,
            detail="Hybrid search is temporarily unavailable.",
        ) from exc

    return success_response(
        data={
            "query": request.query,
            "top_k_per_source": request.top_k,
            "result_count": len(results),
            "results": results,
        },
        message="hybrid candidate retrieval completed",
    )


@router.post(
    "/hybrid/fused",
    response_model=RankFusionSearchResponse,
    summary="Run permission-aware RRF hybrid search",
    description=(
        "Retrieve vector and BM25 candidates from readable documents, then combine "
        "their rankings with Reciprocal Rank Fusion. No LLM is called."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Bearer credentials are missing or invalid.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Authorization or rank-fused retrieval is unavailable.",
        },
    },
)
def fused_hybrid_search(
    request: RankFusionSearchRequest,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    """Retrieve vector and BM25 candidates, then fuse their ranks with RRF."""
    candidate_depth = request.candidate_depth or request.top_k
    allowed_document_ids = _readable_document_ids(current_user.user_id)
    try:
        results = search_fused_hybrid_chunks(
            request.query,
            top_k=request.top_k,
            candidate_depth=candidate_depth,
            rrf_k=request.rrf_k,
            allowed_document_ids=allowed_document_ids,
        )
    except EmbeddingClientError as exc:
        raise HTTPException(
            status_code=503,
            detail="Rank-fused hybrid search is temporarily unavailable.",
        ) from exc

    return success_response(
        data={
            "query": request.query,
            "top_k": request.top_k,
            "candidate_depth": candidate_depth,
            "rrf_k": request.rrf_k,
            "result_count": len(results),
            "results": results,
        },
        message="rank-fused hybrid search completed",
    )
