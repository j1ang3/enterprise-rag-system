import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import delete, inspect, select

from app.db.models import DocumentRecord, UserRecord
from app.db.session import (
    create_database_engine,
    create_session_factory,
    require_test_database_url,
)
from app.services.user_registry import (
    UsernameAlreadyExistsError,
    create_user,
    get_user_by_id,
    get_user_by_username,
)
from app.auth.passwords import hash_password


EXPECTED_USER_COLUMNS = {"user_id", "username", "password_hash", "created_at"}


def _upgrade_test_database(database_url: str) -> Config:
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.attributes["database_url"] = database_url
    command.upgrade(alembic_config, "head")
    return alembic_config


def main() -> int:
    # Strictly requires the explicitly isolated _test URL. There is no fallback.
    test_database_url = require_test_database_url()
    engine = create_database_engine(test_database_url)

    inspector = inspect(engine)
    document_columns_before = {
        column["name"] for column in inspector.get_columns("documents")
    }
    with engine.connect() as connection:
        document_ids_before = set(
            connection.execute(select(DocumentRecord.document_id)).scalars()
        )

    alembic_config = _upgrade_test_database(test_database_url)
    session_factory = create_session_factory(test_database_url)
    inspector = inspect(engine)

    user_columns = {
        column["name"] for column in inspector.get_columns("users")
    }
    if user_columns != EXPECTED_USER_COLUMNS:
        raise RuntimeError("The test users table does not match the W10-T2 schema.")

    unique_constraints = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("users")
    }
    if ("username",) not in unique_constraints:
        raise RuntimeError("PostgreSQL does not enforce unique usernames.")

    document_columns_after = {
        column["name"] for column in inspector.get_columns("documents")
    }
    with engine.connect() as connection:
        current_revision = MigrationContext.configure(connection).get_current_revision()
        document_ids_after = set(
            connection.execute(select(DocumentRecord.document_id)).scalars()
        )
    expected_heads = set(ScriptDirectory.from_config(alembic_config).get_heads())
    if current_revision not in expected_heads:
        raise RuntimeError("The test database is not at the Alembic head revision.")
    if document_columns_after != document_columns_before:
        raise RuntimeError("The W10-T2 migration changed the documents schema.")
    if document_ids_after != document_ids_before:
        raise RuntimeError("The W10-T2 migration changed existing document identities.")

    with session_factory() as session:
        preexisting_user_ids = set(
            session.scalars(select(UserRecord.user_id)).all()
        )

    raw_username = f" W10T2.User.{uuid4()} "
    created = None
    try:
        # create_user uses Session A and closes it after COMMIT.
        created = create_user(
            raw_username,
            session_factory,
            password_hash=hash_password("W10-T2 verifier synthetic password"),
        )
        if created.username != raw_username.strip().lower():
            raise RuntimeError("Username canonicalization changed unexpectedly.")
        if created.created_at.tzinfo is None or created.created_at.utcoffset() is None:
            raise RuntimeError("User created_at is not timezone-aware.")

        # These lookups each use a new Session, proving committed persistence.
        by_id = get_user_by_id(created.user_id, session_factory)
        by_username = get_user_by_username(raw_username.upper(), session_factory)
        if by_id is None or by_username is None:
            raise RuntimeError("The committed user was not readable in a new Session.")
        if by_id.user_id != created.user_id or by_username.user_id != created.user_id:
            raise RuntimeError("The stable user ID changed across persistence lookups.")

        try:
            create_user(
                f"  {created.username.upper()}  ",
                session_factory,
                password_hash=hash_password("W10-T2 verifier duplicate password"),
            )
        except UsernameAlreadyExistsError:
            pass
        else:
            raise RuntimeError("PostgreSQL accepted a duplicate canonical username.")

        if get_user_by_id(created.user_id, session_factory) is None:
            raise RuntimeError("Duplicate rollback damaged subsequent user queries.")
        if get_user_by_id(uuid4(), session_factory) is not None:
            raise RuntimeError("An unknown user ID unexpectedly resolved.")
        if get_user_by_username("w10t2-missing-user", session_factory) is not None:
            raise RuntimeError("An unknown username unexpectedly resolved.")
    finally:
        if created is not None:
            with session_factory() as session:
                with session.begin():
                    session.execute(
                        delete(UserRecord).where(UserRecord.user_id == created.user_id)
                    )

    with session_factory() as session:
        final_user_ids = set(session.scalars(select(UserRecord.user_id)).all())
    if final_user_ids != preexisting_user_ids:
        raise RuntimeError("User verification cleanup left scoped data behind.")

    print(
        "Real PostgreSQL user verification passed: "
        "database=enterprise_rag_test, "
        f"revision={current_revision}, user_columns={len(user_columns)}, "
        "session_a_commit=true, session_b_lookup_by_id=true, "
        "lookup_by_username=true, duplicate_rejected=true, "
        "rollback_reusable=true, documents_unchanged=true, cleanup=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
