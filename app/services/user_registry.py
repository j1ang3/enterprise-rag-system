from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import UserRecord
from app.db.session import get_session_factory


USERNAME_MAX_LENGTH = 64
SessionFactory = Callable[[], Session]
LOGGER = logging.getLogger(__name__)


class UserRegistryError(RuntimeError):
    """Base exception for user identity persistence failures."""


class InvalidUsernameError(UserRegistryError):
    """Raised when a username cannot become a valid canonical identity."""


class UsernameAlreadyExistsError(UserRegistryError):
    """Raised when PostgreSQL rejects a duplicate canonical username."""


class UserRegistryUnavailableError(UserRegistryError):
    """Raised when PostgreSQL cannot serve a user registry operation."""


class InvalidPasswordHashError(UserRegistryError):
    """Raised when credential persistence receives no usable encoded hash."""


@dataclass(frozen=True)
class UserIdentity:
    user_id: UUID
    username: str
    created_at: datetime


@dataclass(frozen=True)
class UserCredential:
    """Internal authentication data; never use this as an API response model."""

    identity: UserIdentity
    password_hash: str


def normalize_username(username: str) -> str:
    if not isinstance(username, str):
        raise InvalidUsernameError("Username must be a string.")
    normalized = username.strip().lower()
    if not normalized:
        raise InvalidUsernameError("Username must not be empty.")
    if len(normalized) > USERNAME_MAX_LENGTH:
        raise InvalidUsernameError(
            f"Username must not exceed {USERNAME_MAX_LENGTH} characters."
        )
    return normalized


def _factory_or_default(session_factory: SessionFactory | None) -> SessionFactory:
    return session_factory or get_session_factory()


def _identity(record: UserRecord) -> UserIdentity:
    return UserIdentity(
        user_id=record.user_id,
        username=record.username,
        created_at=record.created_at,
    )


def _credential(record: UserRecord) -> UserCredential:
    return UserCredential(
        identity=_identity(record),
        password_hash=record.password_hash,
    )


def _log_database_failure(operation: str, exc: Exception) -> None:
    LOGGER.warning(
        "user_registry operation=%s status=failed error_type=%s",
        operation,
        type(exc).__name__,
    )


def create_user(
    username: str,
    session_factory: SessionFactory | None = None,
    *,
    password_hash: str,
) -> UserIdentity:
    canonical_username = normalize_username(username)
    if not isinstance(password_hash, str) or not password_hash.strip():
        raise InvalidPasswordHashError("A non-empty encoded password hash is required.")
    record = UserRecord(
        user_id=uuid4(),
        username=canonical_username,
        password_hash=password_hash,
        created_at=datetime.now(timezone.utc),
    )
    try:
        with _factory_or_default(session_factory)() as session:
            with session.begin():
                session.add(record)
    except IntegrityError as exc:
        _log_database_failure("create", exc)
        raise UsernameAlreadyExistsError(
            "The canonical username already exists."
        ) from exc
    except SQLAlchemyError as exc:
        _log_database_failure("create", exc)
        raise UserRegistryUnavailableError(
            "User identity could not be persisted."
        ) from exc
    return _identity(record)


def get_user_by_id(
    user_id: UUID | str,
    session_factory: SessionFactory | None = None,
) -> UserIdentity | None:
    try:
        canonical_user_id = user_id if isinstance(user_id, UUID) else UUID(str(user_id))
    except (TypeError, ValueError):
        return None

    try:
        with _factory_or_default(session_factory)() as session:
            record = session.get(UserRecord, canonical_user_id)
    except SQLAlchemyError as exc:
        _log_database_failure("get_by_id", exc)
        raise UserRegistryUnavailableError(
            "User identity could not be read."
        ) from exc
    return _identity(record) if record is not None else None


def get_user_by_username(
    username: str,
    session_factory: SessionFactory | None = None,
) -> UserIdentity | None:
    canonical_username = normalize_username(username)
    try:
        with _factory_or_default(session_factory)() as session:
            record = session.scalar(
                select(UserRecord).where(UserRecord.username == canonical_username)
            )
    except SQLAlchemyError as exc:
        _log_database_failure("get_by_username", exc)
        raise UserRegistryUnavailableError(
            "User identity could not be read."
        ) from exc
    return _identity(record) if record is not None else None


def get_user_credential_by_username(
    username: str,
    session_factory: SessionFactory | None = None,
) -> UserCredential | None:
    canonical_username = normalize_username(username)
    try:
        with _factory_or_default(session_factory)() as session:
            record = session.scalar(
                select(UserRecord).where(UserRecord.username == canonical_username)
            )
    except SQLAlchemyError as exc:
        _log_database_failure("get_credential_by_username", exc)
        raise UserRegistryUnavailableError(
            "User credential could not be read."
        ) from exc
    return _credential(record) if record is not None else None
