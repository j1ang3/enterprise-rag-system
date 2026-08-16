from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


class DatabaseConfigurationError(RuntimeError):
    """Raised when a required database URL is absent or unsafe."""


def require_database_url() -> str:
    database_url = settings.database_url.strip()
    if not database_url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured.")
    return database_url


def _database_identity(
    database_url: str,
) -> tuple[str, str | None, str | None, int | None, str | None]:
    parsed = make_url(database_url)
    backend = parsed.drivername.split("+", maxsplit=1)[0]
    return backend, parsed.username, parsed.host, parsed.port, parsed.database


def require_test_database_url() -> str:
    """Return the isolated test URL without ever falling back to DATABASE_URL."""
    test_database_url = settings.test_database_url.strip()
    if not test_database_url:
        raise DatabaseConfigurationError(
            "TEST_DATABASE_URL is required for real database verification."
        )

    production_database_url = settings.database_url.strip()
    if production_database_url and (
        _database_identity(test_database_url)
        == _database_identity(production_database_url)
    ):
        raise DatabaseConfigurationError(
            "TEST_DATABASE_URL must identify a different database from DATABASE_URL."
        )

    parsed = make_url(test_database_url)
    if not parsed.database or not parsed.database.lower().endswith("_test"):
        raise DatabaseConfigurationError(
            "TEST_DATABASE_URL must target a database whose name ends with '_test'."
        )
    return test_database_url


@lru_cache(maxsize=4)
def create_database_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_engine() -> Engine:
    return create_database_engine(require_database_url())


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(
        bind=create_database_engine(database_url),
        expire_on_commit=False,
        class_=Session,
    )


def get_session_factory() -> sessionmaker[Session]:
    return create_session_factory(require_database_url())


@contextmanager
def session_scope(
    session_factory: Callable[[], Session] | None = None,
) -> Iterator[Session]:
    session = (session_factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
