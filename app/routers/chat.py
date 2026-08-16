from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from app.auth.dependencies import get_current_user
from app.db.session import DatabaseConfigurationError
from app.observability.rag_logging import new_request_id
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import ErrorResponse
from app.services.rag_service import answer_question
from app.services.access_control import AccessControlUnavailableError
from app.services.user_registry import UserIdentity
from app.utils.response import success_response

router = APIRouter(prefix="/chat", tags=["chat"])



@router.post(
    "/",
    response_model=ChatResponse,
    summary="Ask the permission-aware RAG system",
    description=(
        "Resolve the authenticated user's readable document IDs, retrieve only "
        "authorized chunks, build context, and return an answer with citations. "
        "LLM unavailability is represented by a successful local_fallback response; "
        "it is not automatically an HTTP 503."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Bearer credentials are missing or invalid.",
        },
        503: {
            "model": ErrorResponse,
            "description": "PostgreSQL-backed document authorization is unavailable.",
        },
    },
)
def chat(
    request: ChatRequest,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    request_id = new_request_id()
    try:
        result = answer_question(
            request.question,
            request.top_k,
            user_id=current_user.user_id,
            retrieval_mode=request.retrieval_mode,
            min_score=request.min_score,
            request_id=request_id,
        )
    except (DatabaseConfigurationError, AccessControlUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document authorization is temporarily unavailable.",
        ) from exc
    # The public response echoes the same opaque identifier even for injected
    # test doubles that predate the observability field.
    result = {**result, "request_id": request_id}

    return success_response(
        data=result,
        message="chat response generated",
    )
