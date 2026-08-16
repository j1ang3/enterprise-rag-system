import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jwt
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, inspect, select
from sqlalchemy.exc import IntegrityError

from app.auth.tokens import JWT_ALGORITHM
from app.core.config import settings
from app.db.models import DocumentRecord, UserRecord
from app.db.session import (
    create_database_engine,
    create_session_factory,
    require_test_database_url,
)
from app.main import app
from app.services.storage_paths import DocumentStoragePaths


TEST_JWT_SECRET = (
    "w10-t4-real-postgresql-test-only-jwt-secret-with-more-than-64-bytes"
)
EXPECTED_DOCUMENT_COLUMNS = {
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


def _document_record(document_id: str, owner_id: UUID | None) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename="integrity.txt",
        file_extension=".txt",
        file_size_bytes=9,
        upload_path=None,
        text_path=None,
        chunk_count=1,
        owner_id=owner_id,
        created_at=None,
    )


def main() -> int:
    test_database_url = require_test_database_url()
    alembic_config = _upgrade_test_database(test_database_url)
    engine = create_database_engine(test_database_url)
    session_factory = create_session_factory(test_database_url)
    inspector = inspect(engine)

    columns = {column["name"] for column in inspector.get_columns("documents")}
    if columns != EXPECTED_DOCUMENT_COLUMNS:
        raise RuntimeError("The test documents table does not match W10-T4.")
    owner_column = next(
        column
        for column in inspector.get_columns("documents")
        if column["name"] == "owner_id"
    )
    if owner_column["nullable"]:
        raise RuntimeError("documents.owner_id is not database-enforced NOT NULL.")

    owner_foreign_keys = [
        foreign_key
        for foreign_key in inspector.get_foreign_keys("documents")
        if foreign_key["constrained_columns"] == ["owner_id"]
    ]
    if len(owner_foreign_keys) != 1:
        raise RuntimeError("documents.owner_id does not have exactly one FK.")
    owner_foreign_key = owner_foreign_keys[0]
    if (
        owner_foreign_key["referred_table"] != "users"
        or owner_foreign_key["referred_columns"] != ["user_id"]
        or owner_foreign_key["options"].get("ondelete") != "RESTRICT"
    ):
        raise RuntimeError("Document ownership FK semantics changed unexpectedly.")

    with engine.connect() as connection:
        current_revision = MigrationContext.configure(connection).get_current_revision()
    if current_revision not in set(
        ScriptDirectory.from_config(alembic_config).get_heads()
    ):
        raise RuntimeError("Test database is not at the Alembic head revision.")

    with session_factory() as session:
        preexisting_user_ids = set(session.scalars(select(UserRecord.user_id)).all())
        preexisting_document_ids = set(
            session.scalars(select(DocumentRecord.document_id)).all()
        )

    created_user_ids: set[UUID] = set()
    uploaded_document_id: str | None = None
    created_document_ids: set[str] = set()
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        storage_paths = DocumentStoragePaths(
            upload_dir=root / "uploads",
            text_dir=root / "texts",
            index_dir=root / "index",
            chunks_file=root / "index" / "chunks.json",
            vectors_file=root / "index" / "vectors.json",
        )
        try:
            with patch.object(
                settings,
                "database_url",
                test_database_url,
            ), patch.object(
                settings,
                "jwt_secret_key",
                SecretStr(TEST_JWT_SECRET),
            ), patch(
                "app.routers.documents.get_document_storage_paths",
                return_value=storage_paths,
            ), patch(
                "app.services.knowledge_base.get_document_storage_paths",
                return_value=storage_paths,
            ), patch(
                "app.services.vector_store.get_document_storage_paths",
                return_value=storage_paths,
            ), patch(
                "app.services.vector_store.embed_text",
                return_value=[0.1, 0.2, 0.3],
            ), patch(
                "app.services.vector_store.embedding_model_for_vector",
                return_value="w10-t4-test-embedding",
            ):
                client = TestClient(app)

                missing_auth = client.post(
                    "/documents/upload",
                    files={"file": ("blocked.txt", b"blocked", "text/plain")},
                )
                invalid_auth = client.post(
                    "/documents/upload",
                    headers={"Authorization": "Bearer malformed"},
                    files={"file": ("blocked.txt", b"blocked", "text/plain")},
                )
                if missing_auth.status_code != 401 or invalid_auth.status_code != 401:
                    raise RuntimeError("Unauthenticated upload was not rejected.")
                if any(
                    path.exists()
                    for path in (
                        storage_paths.upload_dir,
                        storage_paths.text_dir,
                        storage_paths.chunks_file,
                        storage_paths.vectors_file,
                    )
                ):
                    raise RuntimeError("Rejected authentication created durable artifacts.")

                registrations = []
                for label in ("alice", "bob"):
                    response = client.post(
                        "/auth/register",
                        json={
                            "username": f"w10t4-{label}-{uuid4()}",
                            "password": f"W10-T4 {label} synthetic password",
                        },
                    )
                    if response.status_code != 201:
                        raise RuntimeError("Synthetic ownership User registration failed.")
                    user_id = UUID(response.json()["data"]["user_id"])
                    created_user_ids.add(user_id)
                    registrations.append(user_id)
                alice_id, bob_id = registrations

                with session_factory() as session:
                    alice_record = session.get(UserRecord, alice_id)
                login = client.post(
                    "/auth/login",
                    json={
                        "username": alice_record.username,
                        "password": "W10-T4 alice synthetic password",
                    },
                )
                if login.status_code != 200:
                    raise RuntimeError("Synthetic ownership User login failed.")
                access_token = login.json()["data"]["access_token"]
                payload = jwt.decode(
                    access_token,
                    TEST_JWT_SECRET,
                    algorithms=[JWT_ALGORITHM],
                )
                if set(payload) != {"sub", "iat", "exp"}:
                    raise RuntimeError("JWT gained ownership or permission claims.")

                upload = client.post(
                    "/documents/upload",
                    headers={"Authorization": f"Bearer {access_token}"},
                    data={"owner_id": str(bob_id)},
                    files={
                        "file": (
                            "owned-policy.txt",
                            b"Authenticated ownership policy",
                            "text/plain",
                        )
                    },
                )
                if upload.status_code != 200:
                    raise RuntimeError("Authenticated ownership upload failed.")
                uploaded_document_id = upload.json()["data"]["document_id"]
                created_document_ids.add(uploaded_document_id)

                with session_factory() as session:
                    persisted = session.get(DocumentRecord, uploaded_document_id)
                if persisted is None or persisted.owner_id != alice_id:
                    raise RuntimeError(
                        "Upload owner was not derived from current_user.user_id."
                    )
                if persisted.owner_id == bob_id:
                    raise RuntimeError("Client-supplied owner_id spoofing succeeded.")

                anonymous_listing = client.get("/documents/")
                if anonymous_listing.status_code != 401:
                    raise RuntimeError("Anonymous document listing was not rejected.")

                authenticated_listing = client.get(
                    "/documents/",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                listed_ids = {
                    item["document_id"]
                    for item in authenticated_listing.json()["data"]["documents"]
                }
                if (
                    authenticated_listing.status_code != 200
                    or uploaded_document_id not in listed_ids
                ):
                    raise RuntimeError(
                        "Authenticated owner could not list the uploaded document."
                    )

                invalid_owner_id = f"invalid-owner-{uuid4()}"
                created_document_ids.add(invalid_owner_id)
                try:
                    with session_factory() as session:
                        with session.begin():
                            session.add(_document_record(invalid_owner_id, uuid4()))
                except IntegrityError:
                    pass
                else:
                    raise RuntimeError("PostgreSQL accepted a nonexistent owner.")

                ownerless_id = f"ownerless-{uuid4()}"
                created_document_ids.add(ownerless_id)
                try:
                    with session_factory() as session:
                        with session.begin():
                            session.add(_document_record(ownerless_id, None))
                except IntegrityError:
                    pass
                else:
                    raise RuntimeError("PostgreSQL accepted a NULL owner.")

                try:
                    with session_factory() as session:
                        with session.begin():
                            session.execute(
                                delete(UserRecord).where(
                                    UserRecord.user_id == alice_id
                                )
                            )
                except IntegrityError:
                    pass
                else:
                    raise RuntimeError("PostgreSQL deleted an owner with a document.")
        finally:
            with session_factory() as session:
                with session.begin():
                    if created_document_ids:
                        session.execute(
                            delete(DocumentRecord).where(
                                DocumentRecord.document_id.in_(created_document_ids)
                            )
                        )
                    if created_user_ids:
                        session.execute(
                            delete(UserRecord).where(
                                UserRecord.user_id.in_(created_user_ids)
                            )
                        )

    with session_factory() as session:
        final_user_ids = set(session.scalars(select(UserRecord.user_id)).all())
        final_document_ids = set(
            session.scalars(select(DocumentRecord.document_id)).all()
        )
    if final_user_ids != preexisting_user_ids:
        raise RuntimeError("Ownership verifier changed preexisting users.")
    if final_document_ids != preexisting_document_ids:
        raise RuntimeError("Ownership verifier changed preexisting documents.")

    print(
        "Real PostgreSQL ownership verification passed: "
        "database=enterprise_rag_test, "
        f"revision={current_revision}, owner_not_null=true, owner_fk=true, "
        "delete_restrict=true, authenticated_upload=true, "
        "session_b_owner_lookup=true, spoofing_rejected=true, "
        "unauthenticated_no_artifacts=true, invalid_owner_rejected=true, "
        "ownerless_rejected=true, listing_acl_protected=true, exact_cleanup=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
