import sys
from datetime import datetime, timezone
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

from app.auth.passwords import hash_password
from app.db.models import DocumentRecord, UserRecord
from app.db.session import (
    create_database_engine,
    create_session_factory,
    require_test_database_url,
)
from app.services.document_registry import (
    DocumentRegistration,
    DocumentRegistryConflictError,
    backfill_document_registry,
    build_historical_registrations,
    list_registered_documents,
    register_document,
    registered_document_ids,
)
from app.services.knowledge_base import get_all_chunks
from app.services.storage_paths import get_document_storage_paths
from app.services.user_registry import create_user


EXPECTED_COLUMNS = {
    "document_id",
    "original_filename",
    "file_extension",
    "file_size_bytes",
    "upload_path",
    "text_path",
    "chunk_count",
    "owner_id",
    "created_at",
}


def _upgrade_test_database(database_url: str) -> Config:
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.attributes["database_url"] = database_url
    command.upgrade(alembic_config, "head")
    return alembic_config


def main() -> int:
    # This call is intentionally strict: absence, production equality, and a
    # non-_test database name all fail instead of falling back to DATABASE_URL.
    test_database_url = require_test_database_url()
    alembic_config = _upgrade_test_database(test_database_url)
    engine = create_database_engine(test_database_url)
    session_factory = create_session_factory(test_database_url)

    columns = {column["name"] for column in inspect(engine).get_columns("documents")}
    if columns != EXPECTED_COLUMNS:
        raise RuntimeError("The test documents table does not match the W10-T1 schema.")

    with engine.connect() as connection:
        migration_context = MigrationContext.configure(connection)
        current_revision = migration_context.get_current_revision()
    expected_heads = set(ScriptDirectory.from_config(alembic_config).get_heads())
    if current_revision not in expected_heads:
        raise RuntimeError("The test database is not at the Alembic head revision.")

    with session_factory() as session:
        preexisting_user_ids = set(session.scalars(select(UserRecord.user_id)).all())
    owner = create_user(
        f"w10t1-verifier-{uuid4()}",
        session_factory,
        password_hash=hash_password("W10-T1 verifier synthetic password"),
    )
    cleanup_ids: set[str] = set()
    try:
        historical_registrations = build_historical_registrations(
            get_all_chunks(),
            get_document_storage_paths(),
            owner_id=owner.user_id,
        )
        historical_ids = {item.document_id for item in historical_registrations}
        preexisting_ids = registered_document_ids(session_factory)
        backfill_result = backfill_document_registry(
            historical_registrations,
            session_factory,
        )
        after_backfill_ids = registered_document_ids(session_factory)
        cleanup_ids = historical_ids - preexisting_ids

        document_id = f"w10t1-verification-{uuid4()}"
        cleanup_ids.add(document_id)
        registration = DocumentRegistration(
            document_id=document_id,
            original_filename="verification.txt",
            file_extension=".txt",
            file_size_bytes=12,
            upload_path="uploads/verification.txt",
            text_path="texts/verification.txt",
            chunk_count=1,
            owner_id=owner.user_id,
            created_at=datetime.now(timezone.utc),
        )
        if not historical_ids.issubset(after_backfill_ids):
            raise RuntimeError(
                "Test database backfill did not preserve every historical ID."
            )
        register_document(registration, session_factory)
        # register_document committed and closed Session A. This is a distinct
        # Session B read, proving persistence rather than identity-map reuse.
        with session_factory() as session:
            persisted = session.get(DocumentRecord, document_id)
            if persisted is None:
                raise RuntimeError("Committed registry metadata was not readable.")
            if (
                persisted.original_filename != registration.original_filename
                or persisted.file_extension != registration.file_extension
                or persisted.file_size_bytes != registration.file_size_bytes
                or persisted.chunk_count != registration.chunk_count
                or persisted.owner_id != owner.user_id
            ):
                raise RuntimeError("Committed registry metadata values changed.")

        listed_ids = {
            item["document_id"]
            for item in list_registered_documents(session_factory)
        }
        if document_id not in listed_ids or not historical_ids.issubset(listed_ids):
            raise RuntimeError(
                "PostgreSQL-backed application listing omitted verified documents."
            )

        try:
            register_document(registration, session_factory)
        except DocumentRegistryConflictError:
            pass
        else:
            raise RuntimeError("Duplicate document metadata was not rejected.")
    finally:
        with session_factory() as session:
            with session.begin():
                if cleanup_ids:
                    session.execute(
                        delete(DocumentRecord).where(
                            DocumentRecord.document_id.in_(cleanup_ids)
                        )
                    )
                session.execute(
                    delete(UserRecord).where(UserRecord.user_id == owner.user_id)
                )

    with session_factory() as session:
        residue = set(
            session.scalars(
                select(DocumentRecord.document_id).where(
                    DocumentRecord.document_id.in_(cleanup_ids)
                )
            ).all()
        )
    if residue:
        raise RuntimeError("Test verification cleanup left scoped rows behind.")
    with session_factory() as session:
        final_user_ids = set(session.scalars(select(UserRecord.user_id)).all())
    if final_user_ids != preexisting_user_ids:
        raise RuntimeError("Test verification cleanup changed preexisting users.")

    print(
        "Real PostgreSQL verification passed: database=enterprise_rag_test, "
        f"revision={current_revision}, columns={len(columns)}, "
        f"backfill_discovered={backfill_result.discovered}, "
        f"backfill_inserted={backfill_result.inserted}, "
        "commit_read_new_session=true, listing=true, "
        "duplicate_rejected=true, cleanup=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
