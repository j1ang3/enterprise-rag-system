from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.auth.dependencies import get_current_user
from app.core.config import SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE
from app.db.session import DatabaseConfigurationError
from app.schemas.access_control import (
    DocumentShareListResponse,
    GrantDocumentAccessRequest,
    GrantDocumentAccessResponse,
    RevokeDocumentAccessResponse,
)
from app.schemas.documents import (
    DocumentChunksResponse,
    DocumentListResponse,
    DocumentPreviewResponse,
    UploadDocumentResponse,
)
from app.schemas.common import ErrorResponse
from app.services.knowledge_base import get_document_chunks, index_document
from app.services.access_control import (
    ACLManagementForbiddenError,
    AccessControlUnavailableError,
    DocumentNotFoundError,
    DuplicateGrantError,
    GrantNotFoundError,
    OwnerSelfShareError,
    TargetUserNotFoundError,
    can_user_read_document,
    get_readable_document_ids,
    grant_document_read_access,
    list_document_read_grants,
    revoke_document_read_access,
)
from app.services.document_registry import (
    DocumentRegistration,
    DocumentRegistryConflictError,
    DocumentRegistryUnavailableError,
    list_registered_documents,
    register_document,
    verify_document_registry_available,
)
from app.services.embeddings import EmbeddingClientError
from app.services.storage_paths import get_document_storage_paths
from app.services.text_loader import (
    EmptyDocumentError,
    MalformedDocumentError,
    TextExtractionError,
    extract_document,
    make_preview,
)
from app.services.user_registry import UserIdentity
from app.utils.response import success_response


router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)


def _storage_relative_path(path: Path, storage_root: Path) -> str:
    return path.resolve().relative_to(storage_root.resolve()).as_posix()


@router.post(
    "/upload",
    response_model=UploadDocumentResponse,
    summary="Upload and index a document",
    description=(
        "Authenticate the uploader, validate and extract a supported file, create "
        "retrieval chunks, and persist PostgreSQL ownership metadata."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "The uploaded file is invalid."},
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        409: {"model": ErrorResponse, "description": "Document metadata conflicts with an existing record."},
        500: {"model": ErrorResponse, "description": "Local file extraction or persistence failed."},
        503: {"model": ErrorResponse, "description": "Metadata storage or vector indexing is unavailable."},
    },
)
async def upload_document(
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    """
    Upload a document, save it locally, extract its text,
    and save the extracted text for later preview / retrieval.
    """
    original_filename = file.filename

    if not original_filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    file_suffix = Path(original_filename).suffix.lower()

    if file_suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_suffix}. Supported types: {SUPPORTED_EXTENSIONS}"
        )

    document_id = str(uuid4())

    # Keep storage inside the configured upload directory even when a client
    # sends a filename containing path components.
    safe_filename = Path(original_filename).name.replace(" ", "_")
    saved_filename = f"{document_id}_{safe_filename}"
    storage_paths = get_document_storage_paths()
    storage_paths.upload_dir.mkdir(parents=True, exist_ok=True)
    storage_paths.text_dir.mkdir(parents=True, exist_ok=True)
    saved_path = storage_paths.upload_dir / saved_filename

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="Uploaded file is too large.")

    try:
        verify_document_registry_available()
    except (DatabaseConfigurationError, DocumentRegistryUnavailableError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Document metadata storage is temporarily unavailable.",
        ) from exc

    try:
        saved_path.write_bytes(file_bytes)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc

    try:
        extracted_document = extract_document(saved_path)
        extracted_text = extracted_document.text
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=400,
            detail="Document does not contain extractable text.",
        ) from exc
    except MalformedDocumentError as exc:
        raise HTTPException(
            status_code=400,
            detail="Document could not be parsed.",
        ) from exc
    except TextExtractionError as exc:
        raise HTTPException(
            status_code=500,
            detail="Document text extraction is unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to extract text from document.",
        ) from exc

    text_path = storage_paths.text_dir / f"{document_id}.txt"
    try:
        text_path.write_text(extracted_text, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save extracted document text.",
        ) from exc
    try:
        chunks = index_document(
            document_id,
            original_filename,
            extracted_text,
            sections=extracted_document.sections,
        )
    except EmbeddingClientError as exc:
        raise HTTPException(
            status_code=503,
            detail="Document vector indexing is temporarily unavailable.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to persist document index.",
        ) from exc

    try:
        storage_root = storage_paths.upload_dir.parent
        register_document(
            DocumentRegistration(
                document_id=document_id,
                original_filename=original_filename,
                file_extension=file_suffix,
                file_size_bytes=len(file_bytes),
                upload_path=_storage_relative_path(saved_path, storage_root),
                text_path=_storage_relative_path(text_path, storage_root),
                chunk_count=len(chunks),
                owner_id=current_user.user_id,
                created_at=datetime.now(timezone.utc),
            )
        )
    except DocumentRegistryConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Document metadata conflicts with an existing document.",
        ) from exc
    except (DatabaseConfigurationError, DocumentRegistryUnavailableError) as exc:
        # Ingestion files/indexes may already exist at this point. W10-T1 avoids
        # unsafe cross-store cleanup until an atomic ingestion design is added.
        raise HTTPException(
            status_code=503,
            detail="Document was indexed but its metadata could not be persisted.",
        ) from exc

    return success_response(
        data={
            "document_id": document_id,
            "filename": original_filename,
            "content_type": file.content_type,
            "saved_path": str(saved_path),
            "text_path": str(text_path),
            "chunk_count": len(chunks),
            "preview": make_preview(extracted_text),
        },
        message="document uploaded successfully",
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List readable documents",
    description=(
        "Return only documents owned by the authenticated user or explicitly "
        "shared with that user. Filtering happens in PostgreSQL before the response."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        503: {"model": ErrorResponse, "description": "Authorization or metadata storage is unavailable."},
    },
)
def documents(
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    """
    List documents the authenticated user owns or can read through an ACL grant.
    """
    try:
        readable_document_ids = get_readable_document_ids(current_user.user_id)
    except (DatabaseConfigurationError, AccessControlUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document authorization is temporarily unavailable.",
        ) from exc

    try:
        registered_documents = list_registered_documents(
            document_ids=readable_document_ids,
        )
    except (DatabaseConfigurationError, DocumentRegistryUnavailableError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Document metadata storage is temporarily unavailable.",
        ) from exc

    return success_response(
        data={
            "documents": registered_documents,
        },
        message="documents listed",
    )


def _require_document_read_access(document_id: str, user_id: UUID) -> None:
    try:
        is_allowed = can_user_read_document(user_id, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document does not exist.",
        ) from exc
    except (DatabaseConfigurationError, AccessControlUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document authorization is temporarily unavailable.",
        ) from exc

    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to read this document.",
        )


def _raise_acl_http_error(exc: Exception) -> None:
    if isinstance(exc, ACLManagementForbiddenError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the document owner may manage sharing permissions.",
        ) from exc
    if isinstance(exc, (DocumentNotFoundError, TargetUserNotFoundError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (DuplicateGrantError, OwnerSelfShareError)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, GrantNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Document permissions are temporarily unavailable.",
    ) from exc


@router.post(
    "/{document_id}/shares",
    response_model=GrantDocumentAccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Share a document with a reader",
    description=(
        "Allow the document owner to grant an explicit PostgreSQL-backed read ACL "
        "to another existing user."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        403: {"model": ErrorResponse, "description": "Only the owner may manage document shares."},
        404: {"model": ErrorResponse, "description": "The document or target user does not exist."},
        409: {"model": ErrorResponse, "description": "The grant already exists or targets the owner."},
        503: {"model": ErrorResponse, "description": "Document permissions are unavailable."},
    },
)
def grant_document_share(
    document_id: str,
    request: GrantDocumentAccessRequest,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    try:
        grant = grant_document_read_access(
            document_id,
            request.user_id,
            current_user.user_id,
        )
    except (
        DatabaseConfigurationError,
        AccessControlUnavailableError,
        DocumentNotFoundError,
        TargetUserNotFoundError,
        ACLManagementForbiddenError,
        OwnerSelfShareError,
        DuplicateGrantError,
    ) as exc:
        _raise_acl_http_error(exc)
    return success_response(data=grant, message="document access granted")


@router.get(
    "/{document_id}/shares",
    response_model=DocumentShareListResponse,
    summary="List document shares",
    description="Allow the document owner to list explicit read ACL grants.",
    responses={
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        403: {"model": ErrorResponse, "description": "Only the owner may inspect document shares."},
        404: {"model": ErrorResponse, "description": "The document does not exist."},
        503: {"model": ErrorResponse, "description": "Document permissions are unavailable."},
    },
)
def document_shares(
    document_id: str,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    try:
        grants = list_document_read_grants(document_id, current_user.user_id)
    except (
        DatabaseConfigurationError,
        AccessControlUnavailableError,
        DocumentNotFoundError,
        ACLManagementForbiddenError,
    ) as exc:
        _raise_acl_http_error(exc)
    return success_response(
        data={"document_id": document_id, "shares": grants},
        message="document shares listed",
    )


@router.delete(
    "/{document_id}/shares/{user_id}",
    response_model=RevokeDocumentAccessResponse,
    summary="Revoke a document share",
    description=(
        "Allow the document owner to revoke an explicit read ACL. Revocation takes "
        "effect on the reader's next request without issuing a new JWT."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        403: {"model": ErrorResponse, "description": "Only the owner may manage document shares."},
        404: {"model": ErrorResponse, "description": "The document or ACL grant does not exist."},
        503: {"model": ErrorResponse, "description": "Document permissions are unavailable."},
    },
)
def revoke_document_share(
    document_id: str,
    user_id: UUID,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    try:
        revoke_document_read_access(
            document_id,
            user_id,
            current_user.user_id,
        )
    except (
        DatabaseConfigurationError,
        AccessControlUnavailableError,
        DocumentNotFoundError,
        ACLManagementForbiddenError,
        GrantNotFoundError,
    ) as exc:
        _raise_acl_http_error(exc)
    return success_response(
        data={"document_id": document_id, "user_id": user_id},
        message="document access revoked",
    )


@router.get(
    "/{document_id}/preview",
    response_model=DocumentPreviewResponse,
    summary="Preview a readable document",
    description=(
        "Return extracted text only when the authenticated user owns the document "
        "or has an active explicit read grant."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        403: {"model": ErrorResponse, "description": "The user cannot read this document."},
        404: {"model": ErrorResponse, "description": "The document or extracted text does not exist."},
        503: {"model": ErrorResponse, "description": "Document authorization is unavailable."},
    },
)
def preview_document(
    document_id: str,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    """
    Preview extracted text of an uploaded document.
    """
    _require_document_read_access(document_id, current_user.user_id)
    storage_paths = get_document_storage_paths()
    text_path = storage_paths.text_dir / f"{document_id}.txt"

    if not text_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document text not found for document_id: {document_id}"
        )

    text = text_path.read_text(encoding="utf-8")

    return success_response(
        data={
            "document_id": document_id,
            "preview": make_preview(text),
        },
        message="document preview generated",
)


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunksResponse,
    summary="List chunks from a readable document",
    description=(
        "Return raw stored chunk content only when the authenticated user owns the "
        "document or has an active explicit read grant."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Bearer credentials are missing or invalid."},
        403: {"model": ErrorResponse, "description": "The user cannot read this document."},
        404: {"model": ErrorResponse, "description": "The document or chunks do not exist."},
        503: {"model": ErrorResponse, "description": "Document authorization is unavailable."},
    },
)
def document_chunks(
    document_id: str,
    current_user: Annotated[UserIdentity, Depends(get_current_user)],
):
    """
    List stored chunks for an uploaded document.
    """
    _require_document_read_access(document_id, current_user.user_id)
    chunks = get_document_chunks(document_id)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"Document chunks not found for document_id: {document_id}"
        )

    return success_response(
        data={
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunks": chunks,
        },
        message="document chunks listed",
    )
