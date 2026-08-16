from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, delete, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.dependencies import get_current_user
from app.core.config import settings
from app.db.base import Base
from app.db.models import DocumentRecord, UserRecord
from app.main import app
from app.services.document_registry import (
    DocumentOwnershipConflictError,
    DocumentOwnershipMappingError,
    DocumentRegistration,
    DocumentRegistryConflictError,
    assign_document_owners,
    get_document_owner,
    register_document,
)
from app.services.storage_paths import DocumentStoragePaths
from app.services.user_registry import create_user


ENCODED_TEST_HASH = "$argon2id$test-only-encoded-value"
TEST_JWT_SECRET = "w10-t4-test-only-jwt-secret-with-more-than-64-bytes-of-material"


def _sqlite_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


def _registration(document_id: str, owner_id: UUID) -> DocumentRegistration:
    return DocumentRegistration(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        file_extension=".txt",
        file_size_bytes=7,
        upload_path=f"uploads/{document_id}.txt",
        text_path=f"texts/{document_id}.txt",
        chunk_count=1,
        owner_id=owner_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class OwnershipBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.engine = _sqlite_engine()
        # Model metadata represents the final invariant. This test database
        # deliberately represents migration stage 1 so NULL rows can be backfilled.
        owner_column = DocumentRecord.__table__.c.owner_id
        owner_column.nullable = True
        try:
            Base.metadata.create_all(self.engine)
        finally:
            owner_column.nullable = False
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self.alice = create_user(
            "alice",
            self.session_factory,
            password_hash=ENCODED_TEST_HASH,
        )
        self.bob = create_user(
            "bob",
            self.session_factory,
            password_hash=ENCODED_TEST_HASH,
        )

    def tearDown(self):
        self.engine.dispose()

    def _ownerless_document(self, document_id: str) -> None:
        registration = _registration(document_id, self.alice.user_id)
        values = {**registration.__dict__, "owner_id": None}
        with self.session_factory() as session:
            with session.begin():
                session.add(DocumentRecord(**values))

    def test_complete_mapping_assigns_and_same_mapping_is_idempotent(self):
        self._ownerless_document("legacy-a")
        self._ownerless_document("legacy-b")
        mapping = {
            "legacy-a": self.alice.user_id,
            "legacy-b": self.alice.user_id,
        }

        first = assign_document_owners(mapping, self.session_factory)
        second = assign_document_owners(mapping, self.session_factory)

        self.assertEqual((first.assigned, first.unchanged), (2, 0))
        self.assertEqual((second.assigned, second.unchanged), (0, 2))
        self.assertEqual(
            get_document_owner("legacy-a", self.session_factory),
            self.alice.user_id,
        )

    def test_conflicting_mapping_cannot_transfer_owner(self):
        register_document(
            _registration("owned", self.alice.user_id),
            self.session_factory,
        )

        with self.assertRaises(DocumentOwnershipConflictError):
            assign_document_owners(
                {"owned": self.bob.user_id},
                self.session_factory,
            )
        self.assertEqual(
            get_document_owner("owned", self.session_factory),
            self.alice.user_id,
        )

    def test_unknown_and_incomplete_mappings_are_rejected(self):
        self._ownerless_document("legacy-a")
        self._ownerless_document("legacy-b")

        invalid_mappings = (
            {"missing-document": self.alice.user_id},
            {"legacy-a": uuid4()},
            {"legacy-a": self.alice.user_id},
        )
        for mapping in invalid_mappings:
            with self.subTest(mapping_size=len(mapping)):
                with self.assertRaises(DocumentOwnershipMappingError):
                    assign_document_owners(mapping, self.session_factory)


class OwnershipFinalIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.engine = _sqlite_engine()
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self.owner = create_user(
            "owner",
            self.session_factory,
            password_hash=ENCODED_TEST_HASH,
        )

    def tearDown(self):
        self.engine.dispose()

    def test_foreign_key_rejects_unknown_owner(self):
        with self.assertRaises(DocumentRegistryConflictError):
            register_document(
                _registration("invalid-owner", uuid4()),
                self.session_factory,
            )

    def test_not_null_rejects_ownerless_document(self):
        values = {**_registration("ownerless", self.owner.user_id).__dict__}
        values["owner_id"] = None
        with self.assertRaises(IntegrityError):
            with self.session_factory() as session:
                with session.begin():
                    session.add(DocumentRecord(**values))

    def test_user_delete_is_restricted_while_document_is_owned(self):
        register_document(
            _registration("owned", self.owner.user_id),
            self.session_factory,
        )
        with self.assertRaises(IntegrityError):
            with self.session_factory() as session:
                with session.begin():
                    session.execute(
                        delete(UserRecord).where(
                            UserRecord.user_id == self.owner.user_id
                        )
                    )


class UploadAuthenticationBoundaryTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides.pop(get_current_user, None)
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.paths = DocumentStoragePaths(
            upload_dir=root / "uploads",
            text_dir=root / "texts",
            index_dir=root / "index",
            chunks_file=root / "index" / "chunks.json",
            vectors_file=root / "index" / "vectors.json",
        )
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_current_user, None)
        self.tempdir.cleanup()

    def test_missing_and_invalid_tokens_fail_before_durable_writes(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.paths,
        ), patch.object(
            settings,
            "jwt_secret_key",
            SecretStr(TEST_JWT_SECRET),
        ):
            missing = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )
            invalid = self.client.post(
                "/documents/upload",
                headers={"Authorization": "Bearer malformed"},
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertFalse(self.paths.upload_dir.exists())
        self.assertFalse(self.paths.text_dir.exists())
        self.assertFalse(self.paths.chunks_file.exists())
        self.assertFalse(self.paths.vectors_file.exists())


if __name__ == "__main__":
    unittest.main()
