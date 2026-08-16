from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from uuid import UUID

from sqlalchemy import select, union
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import DocumentACLRecord, DocumentRecord, UserRecord
from app.db.session import get_session_factory


SessionFactory = Callable[[], Session]
LOGGER = logging.getLogger(__name__)


class AccessControlError(RuntimeError):
    """Base exception for document ACL operations."""


class AccessControlUnavailableError(AccessControlError):
    """Raised when PostgreSQL cannot serve an authorization operation."""


class DocumentNotFoundError(AccessControlError):
    """Raised when an authorization operation references no document."""


class TargetUserNotFoundError(AccessControlError):
    """Raised when a share recipient does not exist."""


class ACLManagementForbiddenError(AccessControlError):
    """Raised when a non-owner attempts to manage document shares."""


class OwnerSelfShareError(AccessControlError):
    """Raised when an owner attempts to create a redundant ACL row."""


class DuplicateGrantError(AccessControlError):
    """Raised when the requested explicit read grant already exists."""


class GrantNotFoundError(AccessControlError):
    """Raised when the requested explicit read grant does not exist."""


@dataclass(frozen=True)
class DocumentGrant:
    document_id: str
    user_id: UUID
    username: str
    created_at: datetime


@dataclass(frozen=True)
class RetrievalAuthorizationContext:
    user_id: UUID
    readable_document_ids: frozenset[str]


def _factory_or_default(session_factory: SessionFactory | None) -> SessionFactory:
    return session_factory or get_session_factory()


def _log_database_failure(operation: str, exc: Exception) -> None:
    LOGGER.warning(
        "access_control operation=%s status=failed error_type=%s",
        operation,
        type(exc).__name__,
    )


def _owned_document(
    session: Session,
    document_id: str,
    acting_user_id: UUID,
) -> DocumentRecord:
    document = session.get(DocumentRecord, document_id)
    if document is None:
        raise DocumentNotFoundError("Document does not exist.")
    if document.owner_id != acting_user_id:
        raise ACLManagementForbiddenError(
            "Only the document owner may manage sharing permissions."
        )
    return document


def can_user_read_document(
    user_id: UUID,
    document_id: str,
    session_factory: SessionFactory | None = None,
) -> bool:
    """Apply default-deny read authorization without retrieval integration."""
    try:
        with _factory_or_default(session_factory)() as session:
            document = session.get(DocumentRecord, document_id)
            if document is None:
                raise DocumentNotFoundError("Document does not exist.")
            if document.owner_id == user_id:
                return True
            grant = session.get(DocumentACLRecord, (document_id, user_id))
            return grant is not None
    except DocumentNotFoundError:
        raise
    except SQLAlchemyError as exc:
        _log_database_failure("can_read", exc)
        raise AccessControlUnavailableError(
            "Document permissions could not be read."
        ) from exc


def get_readable_document_ids(
    user_id: UUID,
    session_factory: SessionFactory | None = None,
) -> frozenset[str]:
    """Resolve owned and explicitly shared documents in one SQL query."""
    owned = select(DocumentRecord.document_id).where(
        DocumentRecord.owner_id == user_id
    )
    shared = select(DocumentACLRecord.document_id).where(
        DocumentACLRecord.user_id == user_id
    )
    try:
        with _factory_or_default(session_factory)() as session:
            return frozenset(session.scalars(union(owned, shared)).all())
    except SQLAlchemyError as exc:
        _log_database_failure("readable_documents", exc)
        raise AccessControlUnavailableError(
            "Readable document permissions could not be resolved."
        ) from exc


def resolve_retrieval_authorization(
    user_id: UUID,
    session_factory: SessionFactory | None = None,
) -> RetrievalAuthorizationContext:
    return RetrievalAuthorizationContext(
        user_id=user_id,
        readable_document_ids=get_readable_document_ids(user_id, session_factory),
    )


def grant_document_read_access(
    document_id: str,
    target_user_id: UUID,
    acting_user_id: UUID,
    session_factory: SessionFactory | None = None,
) -> DocumentGrant:
    try:
        with _factory_or_default(session_factory)() as session:
            with session.begin():
                document = _owned_document(session, document_id, acting_user_id)
                target = session.get(UserRecord, target_user_id)
                if target is None:
                    raise TargetUserNotFoundError("Share recipient does not exist.")
                if target.user_id == document.owner_id:
                    raise OwnerSelfShareError(
                        "The owner already has implicit document access."
                    )
                if session.get(
                    DocumentACLRecord,
                    (document_id, target_user_id),
                ) is not None:
                    raise DuplicateGrantError("Read access is already granted.")
                created_at = datetime.now(timezone.utc)
                session.add(
                    DocumentACLRecord(
                        document_id=document_id,
                        user_id=target_user_id,
                        created_at=created_at,
                    )
                )
        return DocumentGrant(
            document_id=document_id,
            user_id=target.user_id,
            username=target.username,
            created_at=created_at,
        )
    except (
        DocumentNotFoundError,
        TargetUserNotFoundError,
        ACLManagementForbiddenError,
        OwnerSelfShareError,
        DuplicateGrantError,
    ):
        raise
    except IntegrityError as exc:
        _log_database_failure("grant_conflict", exc)
        raise DuplicateGrantError(
            "Read access could not be granted because it conflicts with current data."
        ) from exc
    except SQLAlchemyError as exc:
        _log_database_failure("grant", exc)
        raise AccessControlUnavailableError(
            "Document permissions could not be updated."
        ) from exc


def revoke_document_read_access(
    document_id: str,
    target_user_id: UUID,
    acting_user_id: UUID,
    session_factory: SessionFactory | None = None,
) -> None:
    try:
        with _factory_or_default(session_factory)() as session:
            with session.begin():
                _owned_document(session, document_id, acting_user_id)
                grant = session.get(
                    DocumentACLRecord,
                    (document_id, target_user_id),
                )
                if grant is None:
                    raise GrantNotFoundError("Read access grant does not exist.")
                session.delete(grant)
    except (
        DocumentNotFoundError,
        ACLManagementForbiddenError,
        GrantNotFoundError,
    ):
        raise
    except SQLAlchemyError as exc:
        _log_database_failure("revoke", exc)
        raise AccessControlUnavailableError(
            "Document permissions could not be updated."
        ) from exc


def list_document_read_grants(
    document_id: str,
    acting_user_id: UUID,
    session_factory: SessionFactory | None = None,
) -> list[DocumentGrant]:
    try:
        with _factory_or_default(session_factory)() as session:
            _owned_document(session, document_id, acting_user_id)
            rows = session.execute(
                select(DocumentACLRecord, UserRecord.username)
                .join(UserRecord, UserRecord.user_id == DocumentACLRecord.user_id)
                .where(DocumentACLRecord.document_id == document_id)
                .order_by(UserRecord.username, DocumentACLRecord.user_id)
            ).all()
            return [
                DocumentGrant(
                    document_id=grant.document_id,
                    user_id=grant.user_id,
                    username=username,
                    created_at=grant.created_at,
                )
                for grant, username in rows
            ]
    except (DocumentNotFoundError, ACLManagementForbiddenError):
        raise
    except SQLAlchemyError as exc:
        _log_database_failure("list", exc)
        raise AccessControlUnavailableError(
            "Document permissions could not be read."
        ) from exc
