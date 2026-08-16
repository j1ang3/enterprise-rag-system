import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, inspect, select
from sqlalchemy.exc import IntegrityError

from app.auth.dependencies import get_current_user
from app.db.models import DocumentACLRecord, DocumentRecord, UserRecord
from app.db.session import (
    create_database_engine,
    create_session_factory,
    require_test_database_url,
)
from app.main import app
from app.services.access_control import can_user_read_document
from app.services.user_registry import UserIdentity


ENCODED_TEST_HASH = "$argon2id$w10-t5-test-only-encoded-value"


def _upgrade_test_database(database_url: str) -> Config:
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.attributes["database_url"] = database_url
    command.upgrade(alembic_config, "head")
    return alembic_config


def _document(document_id: str, owner_id: UUID) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        file_extension=".txt",
        file_size_bytes=8,
        upload_path=None,
        text_path=None,
        chunk_count=1,
        owner_id=owner_id,
        created_at=datetime.now(timezone.utc),
    )


def main() -> int:
    test_database_url = require_test_database_url()
    engine = create_database_engine(test_database_url)
    session_factory = create_session_factory(test_database_url)

    with session_factory() as session:
        documents_before = {
            record.document_id: record.owner_id
            for record in session.scalars(select(DocumentRecord)).all()
        }

    alembic_config = _upgrade_test_database(test_database_url)
    inspector = inspect(engine)
    with engine.connect() as connection:
        current_revision = MigrationContext.configure(connection).get_current_revision()
    if current_revision not in set(
        ScriptDirectory.from_config(alembic_config).get_heads()
    ):
        raise RuntimeError("Test database is not at the Alembic head revision.")

    inspected_columns = inspector.get_columns("document_acl")
    columns = {column["name"] for column in inspected_columns}
    if columns != {"document_id", "user_id", "created_at"}:
        raise RuntimeError("document_acl columns do not match W10-T5.")
    created_at_column = next(
        column for column in inspected_columns if column["name"] == "created_at"
    )
    if (
        created_at_column["nullable"]
        or not getattr(created_at_column["type"], "timezone", False)
        or created_at_column["default"] is None
    ):
        raise RuntimeError("document_acl.created_at is not server-defaulted TIMESTAMPTZ.")
    primary_key = inspector.get_pk_constraint("document_acl")
    if primary_key["constrained_columns"] != ["document_id", "user_id"]:
        raise RuntimeError("document_acl does not use the required composite key.")
    foreign_keys = {
        (
            tuple(foreign_key["constrained_columns"]),
            foreign_key["referred_table"],
            tuple(foreign_key["referred_columns"]),
            foreign_key["options"].get("ondelete"),
        )
        for foreign_key in inspector.get_foreign_keys("document_acl")
    }
    if foreign_keys != {
        (("document_id",), "documents", ("document_id",), "CASCADE"),
        (("user_id",), "users", ("user_id",), "CASCADE"),
    }:
        raise RuntimeError("document_acl foreign-key semantics changed unexpectedly.")

    with session_factory() as session:
        if session.scalar(select(func.count()).select_from(DocumentACLRecord)) != 0:
            raise RuntimeError("Test ACL table was not empty before synthetic verification.")
        documents_after_migration = {
            record.document_id: record.owner_id
            for record in session.scalars(select(DocumentRecord)).all()
        }
    if documents_after_migration != documents_before:
        raise RuntimeError("ACL migration changed existing document ownership.")

    synthetic_suffix = uuid4().hex
    user_ids = {name: uuid4() for name in ("alice", "bob", "carol")}
    document_id = f"w10t5-document-{synthetic_suffix}"
    created_at = datetime.now(timezone.utc)
    try:
        with session_factory() as session:
            with session.begin():
                for name, user_id in user_ids.items():
                    session.add(
                        UserRecord(
                            user_id=user_id,
                            username=f"w10t5-{name}-{synthetic_suffix}",
                            password_hash=ENCODED_TEST_HASH,
                            created_at=created_at,
                        )
                    )
                # These models intentionally have no ORM relationship; flush the
                # FK principals before inserting the owned document.
                session.flush()
                session.add(_document(document_id, user_ids["alice"]))

        if not can_user_read_document(
            user_ids["alice"], document_id, session_factory
        ):
            raise RuntimeError("Owner implicit access was denied.")
        if can_user_read_document(user_ids["bob"], document_id, session_factory):
            raise RuntimeError("Default deny failed for a non-owner.")

        identities = {
            name: UserIdentity(
                user_id=user_id,
                username=f"w10t5-{name}-{synthetic_suffix}",
                created_at=created_at,
            )
            for name, user_id in user_ids.items()
        }
        with patch(
            "app.services.access_control.get_session_factory",
            return_value=session_factory,
        ):
            client = TestClient(app)
            unauthenticated = client.post(
                f"/documents/{document_id}/shares",
                json={"user_id": str(user_ids["bob"])},
            )
            if unauthenticated.status_code != 401:
                raise RuntimeError("Unauthenticated ACL management was not rejected.")

            app.dependency_overrides[get_current_user] = lambda: identities["alice"]
            granted = client.post(
                f"/documents/{document_id}/shares",
                json={"user_id": str(user_ids["bob"])},
            )
            duplicate = client.post(
                f"/documents/{document_id}/shares",
                json={"user_id": str(user_ids["bob"])},
            )
            self_share = client.post(
                f"/documents/{document_id}/shares",
                json={"user_id": str(user_ids["alice"])},
            )
            invalid_target = client.post(
                f"/documents/{document_id}/shares",
                json={"user_id": str(uuid4())},
            )
            invalid_document = client.post(
                f"/documents/missing-{synthetic_suffix}/shares",
                json={"user_id": str(user_ids["bob"])},
            )
            listed = client.get(f"/documents/{document_id}/shares")
            if [
                granted.status_code,
                duplicate.status_code,
                self_share.status_code,
                invalid_target.status_code,
                invalid_document.status_code,
                listed.status_code,
            ] != [201, 409, 409, 404, 404, 200]:
                raise RuntimeError("Owner ACL API contract verification failed.")
            if listed.json()["data"]["shares"][0]["user_id"] != str(
                user_ids["bob"]
            ):
                raise RuntimeError("Grant list did not return the expected recipient.")
            if "password_hash" in listed.text:
                raise RuntimeError("Grant list exposed credential material.")

            app.dependency_overrides[get_current_user] = lambda: identities["bob"]
            non_owner_responses = (
                client.post(
                    f"/documents/{document_id}/shares",
                    json={"user_id": str(user_ids["carol"])},
                ),
                client.get(f"/documents/{document_id}/shares"),
                client.delete(
                    f"/documents/{document_id}/shares/{user_ids['carol']}"
                ),
            )
            if [response.status_code for response in non_owner_responses] != [403] * 3:
                raise RuntimeError("Non-owner ACL management was not denied.")

            app.dependency_overrides[get_current_user] = lambda: identities["alice"]
            revoked = client.delete(
                f"/documents/{document_id}/shares/{user_ids['bob']}"
            )
            missing_revoke = client.delete(
                f"/documents/{document_id}/shares/{user_ids['bob']}"
            )
            if [revoked.status_code, missing_revoke.status_code] != [200, 404]:
                raise RuntimeError("Revoke API contract verification failed.")
        app.dependency_overrides.pop(get_current_user, None)

        if can_user_read_document(user_ids["bob"], document_id, session_factory):
            raise RuntimeError("Revocation did not take effect immediately.")

        with session_factory() as session:
            owner_acl = session.get(
                DocumentACLRecord,
                (document_id, user_ids["alice"]),
            )
        if owner_acl is not None:
            raise RuntimeError("Owner access was duplicated into the ACL table.")

        with session_factory() as session:
            with session.begin():
                session.add(
                    DocumentACLRecord(
                        document_id=document_id,
                        user_id=user_ids["bob"],
                        created_at=created_at,
                    )
                )
        try:
            with session_factory() as session:
                with session.begin():
                    session.add(
                        DocumentACLRecord(
                            document_id=document_id,
                            user_id=user_ids["bob"],
                            created_at=created_at,
                        )
                    )
        except IntegrityError:
            pass
        else:
            raise RuntimeError("PostgreSQL accepted a duplicate ACL grant.")

        for invalid_row in (
            DocumentACLRecord(
                document_id=f"missing-{synthetic_suffix}",
                user_id=user_ids["carol"],
                created_at=created_at,
            ),
            DocumentACLRecord(
                document_id=document_id,
                user_id=uuid4(),
                created_at=created_at,
            ),
        ):
            try:
                with session_factory() as session:
                    with session.begin():
                        session.add(invalid_row)
            except IntegrityError:
                pass
            else:
                raise RuntimeError("PostgreSQL accepted an orphan ACL grant.")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(DocumentACLRecord).where(
                        DocumentACLRecord.document_id == document_id
                    )
                )
                session.execute(
                    delete(DocumentRecord).where(
                        DocumentRecord.document_id == document_id
                    )
                )
                session.execute(
                    delete(UserRecord).where(
                        UserRecord.user_id.in_(set(user_ids.values()))
                    )
                )

    with session_factory() as session:
        final_documents = {
            record.document_id: record.owner_id
            for record in session.scalars(select(DocumentRecord)).all()
        }
        final_acl_count = session.scalar(
            select(func.count()).select_from(DocumentACLRecord)
        )
    if final_documents != documents_before or final_acl_count != 0:
        raise RuntimeError("ACL verifier did not restore the preexisting test state.")

    print(
        "Real PostgreSQL ACL verification passed: "
        "database=enterprise_rag_test, "
        f"revision={current_revision}, acl_bootstrap_rows=0, "
        "composite_pk=true, foreign_keys=true, default_deny=true, "
        "owner_implicit_allow=true, grant_round_trip=true, revoke_immediate=true, "
        "duplicate_rejected=true, owner_self_share_rejected=true, "
        "owner_only_management=true, api_authentication=true, exact_cleanup=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
