from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import DocumentRecord, UserRecord
from app.db.session import get_session_factory
from app.services.storage_paths import DocumentStoragePaths


SessionFactory = Callable[[], Session]
LOGGER = logging.getLogger(__name__)


class DocumentRegistryError(RuntimeError):
    """Base exception for safe document-registry failures."""


class DocumentRegistryUnavailableError(DocumentRegistryError):
    """Raised when PostgreSQL cannot serve a registry operation."""


class DocumentRegistryConflictError(DocumentRegistryError):
    """Raised when one document ID maps to incompatible metadata."""


class HistoricalRegistryDataError(DocumentRegistryError):
    """Raised when historical chunk metadata is internally inconsistent."""


class DocumentOwnershipMappingError(DocumentRegistryError):
    """Raised when an explicit ownership bootstrap mapping is invalid."""


class DocumentOwnershipConflictError(DocumentRegistryError):
    """Raised when bootstrap would silently transfer an existing owner."""


@dataclass(frozen=True)
class DocumentRegistration:
    document_id: str
    original_filename: str
    file_extension: str
    file_size_bytes: int | None
    upload_path: str | None
    text_path: str | None
    chunk_count: int
    owner_id: UUID
    created_at: datetime | None


@dataclass(frozen=True)
class BackfillResult:
    discovered: int
    inserted: int
    skipped_existing: int


@dataclass(frozen=True)
class OwnershipBackfillResult:
    mapped: int
    assigned: int
    unchanged: int


def _factory_or_default(session_factory: SessionFactory | None) -> SessionFactory:
    return session_factory or get_session_factory()


def _log_database_failure(operation: str, exc: Exception) -> None:
    LOGGER.warning(
        "document_registry operation=%s status=failed error_type=%s",
        operation,
        type(exc).__name__,
    )


def verify_document_registry_available(
    session_factory: SessionFactory | None = None,
) -> None:
    """Fail before ingestion creates durable files when PostgreSQL is unavailable."""
    try:
        with _factory_or_default(session_factory)() as session:
            # Checking the table catches a missing migration before file writes.
            session.execute(select(DocumentRecord.document_id).limit(1))
    except SQLAlchemyError as exc:
        _log_database_failure("availability_check", exc)
        raise DocumentRegistryUnavailableError(
            "Document metadata storage is unavailable."
        ) from exc


def register_document(
    registration: DocumentRegistration,
    session_factory: SessionFactory | None = None,
) -> None:
    try:
        with _factory_or_default(session_factory)() as session:
            with session.begin():
                session.add(DocumentRecord(**registration.__dict__))
    except IntegrityError as exc:
        _log_database_failure("insert", exc)
        raise DocumentRegistryConflictError(
            "Document metadata already exists for this document ID."
        ) from exc
    except SQLAlchemyError as exc:
        _log_database_failure("insert", exc)
        raise DocumentRegistryUnavailableError(
            "Document metadata could not be persisted."
        ) from exc


def list_registered_documents(
    session_factory: SessionFactory | None = None,
    *,
    document_ids: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    if document_ids is not None and not document_ids:
        return []

    query = select(DocumentRecord)
    if document_ids is not None:
        # Apply authorization in SQL so inaccessible metadata never leaves the
        # persistence boundary for the API layer to filter later.
        query = query.where(DocumentRecord.document_id.in_(tuple(document_ids)))
    query = query.order_by(
        DocumentRecord.created_at.desc().nullslast(),
        DocumentRecord.document_id,
    )

    try:
        with _factory_or_default(session_factory)() as session:
            records = session.scalars(query).all()
    except SQLAlchemyError as exc:
        _log_database_failure("list", exc)
        raise DocumentRegistryUnavailableError(
            "Document metadata could not be read."
        ) from exc

    return [
        {
            "document_id": record.document_id,
            "filename": record.original_filename,
            "chunk_count": record.chunk_count,
            "created_at": (
                record.created_at.isoformat() if record.created_at is not None else None
            ),
        }
        for record in records
    ]


def _parse_chunk_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_to_storage(path: Path, storage_paths: DocumentStoragePaths) -> str:
    storage_root = storage_paths.upload_dir.parent.resolve()
    return path.resolve().relative_to(storage_root).as_posix()


def _find_upload_path(
    document_id: str,
    filename: str,
    storage_paths: DocumentStoragePaths,
) -> Path | None:
    expected = storage_paths.upload_dir / (
        f"{document_id}_{Path(filename).name.replace(' ', '_')}"
    )
    if expected.is_file():
        return expected

    prefix = f"{document_id}_"
    candidates = (
        sorted(
            path
            for path in storage_paths.upload_dir.iterdir()
            if path.is_file() and path.name.startswith(prefix)
        )
        if storage_paths.upload_dir.exists()
        else []
    )
    if len(candidates) > 1:
        raise HistoricalRegistryDataError(
            f"Multiple upload files match historical document ID {document_id!r}."
        )
    return candidates[0] if candidates else None


def build_historical_registrations(
    chunks: Iterable[Mapping[str, Any]],
    storage_paths: DocumentStoragePaths,
    *,
    owner_id: UUID,
) -> list[DocumentRegistration]:
    """Build one conservative registry row per document represented by chunks."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        document_id = chunk.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise HistoricalRegistryDataError(
                "A historical chunk is missing a stable document_id."
            )
        grouped[document_id].append(chunk)

    registrations: list[DocumentRegistration] = []
    for document_id, document_chunks in sorted(grouped.items()):
        filenames = {
            chunk.get("filename")
            for chunk in document_chunks
            if isinstance(chunk.get("filename"), str) and chunk.get("filename")
        }
        if len(filenames) != 1:
            raise HistoricalRegistryDataError(
                f"Historical chunks disagree on filename for {document_id!r}."
            )
        filename = filenames.pop()
        upload_file = _find_upload_path(document_id, filename, storage_paths)
        extracted_text_file = storage_paths.text_dir / f"{document_id}.txt"
        timestamps = [
            timestamp
            for timestamp in (
                _parse_chunk_timestamp(chunk.get("created_at"))
                for chunk in document_chunks
            )
            if timestamp is not None
        ]
        registrations.append(
            DocumentRegistration(
                document_id=document_id,
                original_filename=filename,
                file_extension=Path(filename).suffix.lower(),
                file_size_bytes=(upload_file.stat().st_size if upload_file else None),
                upload_path=(
                    _relative_to_storage(upload_file, storage_paths)
                    if upload_file
                    else None
                ),
                text_path=(
                    _relative_to_storage(extracted_text_file, storage_paths)
                    if extracted_text_file.is_file()
                    else None
                ),
                chunk_count=len(document_chunks),
                owner_id=owner_id,
                created_at=min(timestamps) if timestamps else None,
            )
        )
    return registrations


def _conflicting_fields(
    existing: DocumentRecord,
    incoming: DocumentRegistration,
) -> list[str]:
    fields = (
        "original_filename",
        "file_extension",
        "chunk_count",
        "owner_id",
    )
    conflicts = [
        field
        for field in fields
        if getattr(existing, field) != getattr(incoming, field)
    ]
    for field in ("file_size_bytes", "upload_path", "text_path"):
        old_value = getattr(existing, field)
        new_value = getattr(incoming, field)
        if old_value is not None and new_value is not None and old_value != new_value:
            conflicts.append(field)
    return conflicts


def backfill_document_registry(
    registrations: Iterable[DocumentRegistration],
    session_factory: SessionFactory | None = None,
) -> BackfillResult:
    candidates = list(registrations)
    if len({item.document_id for item in candidates}) != len(candidates):
        raise HistoricalRegistryDataError(
            "Backfill candidates contain duplicate document IDs."
        )

    inserted = 0
    skipped = 0
    try:
        with _factory_or_default(session_factory)() as session:
            with session.begin():
                existing_by_id = {
                    record.document_id: record
                    for record in session.scalars(
                        select(DocumentRecord).where(
                            DocumentRecord.document_id.in_(
                                [item.document_id for item in candidates]
                            )
                        )
                    ).all()
                }
                for candidate in candidates:
                    existing = existing_by_id.get(candidate.document_id)
                    if existing is not None:
                        conflicts = _conflicting_fields(existing, candidate)
                        if conflicts:
                            raise DocumentRegistryConflictError(
                                "Historical metadata conflicts for document ID "
                                f"{candidate.document_id!r}: {', '.join(conflicts)}."
                            )
                        skipped += 1
                        continue
                    session.add(DocumentRecord(**candidate.__dict__))
                    inserted += 1
    except DocumentRegistryConflictError:
        raise
    except SQLAlchemyError as exc:
        _log_database_failure("backfill", exc)
        raise DocumentRegistryUnavailableError(
            "Historical document metadata could not be persisted."
        ) from exc

    return BackfillResult(
        discovered=len(candidates),
        inserted=inserted,
        skipped_existing=skipped,
    )


def registered_document_ids(
    session_factory: SessionFactory | None = None,
) -> set[str]:
    try:
        with _factory_or_default(session_factory)() as session:
            return set(session.scalars(select(DocumentRecord.document_id)).all())
    except SQLAlchemyError as exc:
        _log_database_failure("verify_ids", exc)
        raise DocumentRegistryUnavailableError(
            "Document registry verification failed."
        ) from exc


def _canonical_owner_mapping(
    mapping: Mapping[str, UUID | str],
) -> dict[str, UUID]:
    canonical: dict[str, UUID] = {}
    for document_id, owner_id in mapping.items():
        if not isinstance(document_id, str) or not document_id:
            raise DocumentOwnershipMappingError(
                "Every ownership mapping requires a non-empty document ID."
            )
        try:
            canonical[document_id] = (
                owner_id if isinstance(owner_id, UUID) else UUID(str(owner_id))
            )
        except (TypeError, ValueError) as exc:
            raise DocumentOwnershipMappingError(
                "Every ownership mapping requires a valid user UUID."
            ) from exc
    if not canonical:
        raise DocumentOwnershipMappingError(
            "At least one explicit ownership mapping is required."
        )
    return canonical


def assign_document_owners(
    mapping: Mapping[str, UUID | str],
    session_factory: SessionFactory | None = None,
    *,
    require_complete: bool = True,
) -> OwnershipBackfillResult:
    """Apply explicit bootstrap ownership without supporting owner transfer."""
    canonical = _canonical_owner_mapping(mapping)
    try:
        with _factory_or_default(session_factory)() as session:
            with session.begin():
                documents = {
                    record.document_id: record
                    for record in session.scalars(select(DocumentRecord)).all()
                }
                document_ids = set(documents)
                mapping_ids = set(canonical)
                unknown_documents = mapping_ids - document_ids
                missing_documents = document_ids - mapping_ids
                if unknown_documents:
                    raise DocumentOwnershipMappingError(
                        "Ownership mapping contains unknown document IDs."
                    )
                if require_complete and missing_documents:
                    raise DocumentOwnershipMappingError(
                        "Ownership mapping does not cover every document."
                    )

                owner_ids = set(canonical.values())
                existing_owner_ids = set(
                    session.scalars(
                        select(UserRecord.user_id).where(
                            UserRecord.user_id.in_(owner_ids)
                        )
                    ).all()
                )
                if existing_owner_ids != owner_ids:
                    raise DocumentOwnershipMappingError(
                        "Ownership mapping references unknown users."
                    )

                assigned = 0
                unchanged = 0
                for document_id, owner_id in canonical.items():
                    document = documents[document_id]
                    if document.owner_id is None:
                        document.owner_id = owner_id
                        assigned += 1
                    elif document.owner_id == owner_id:
                        unchanged += 1
                    else:
                        raise DocumentOwnershipConflictError(
                            "Ownership bootstrap cannot transfer an existing owner."
                        )
    except (DocumentOwnershipMappingError, DocumentOwnershipConflictError):
        raise
    except SQLAlchemyError as exc:
        _log_database_failure("ownership_backfill", exc)
        raise DocumentRegistryUnavailableError(
            "Document ownership could not be persisted."
        ) from exc
    return OwnershipBackfillResult(
        mapped=len(canonical),
        assigned=assigned,
        unchanged=unchanged,
    )


def get_document_owner(
    document_id: str,
    session_factory: SessionFactory | None = None,
) -> UUID | None:
    try:
        with _factory_or_default(session_factory)() as session:
            record = session.get(DocumentRecord, document_id)
            return record.owner_id if record is not None else None
    except SQLAlchemyError as exc:
        _log_database_failure("get_owner", exc)
        raise DocumentRegistryUnavailableError(
            "Document ownership could not be read."
        ) from exc
