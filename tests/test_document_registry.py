import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.services.document_registry import (
    DocumentRegistration,
    DocumentRegistryConflictError,
    DocumentRegistryUnavailableError,
    backfill_document_registry,
    build_historical_registrations,
    list_registered_documents,
    register_document,
    registered_document_ids,
    verify_document_registry_available,
)
from app.services.storage_paths import DocumentStoragePaths


class DocumentRegistryTests(unittest.TestCase):
    OWNER_ID = UUID("00000000-0000-0000-0000-000000000101")

    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )

    def tearDown(self):
        self.engine.dispose()

    def registration(self, document_id: str = "eval-readme") -> DocumentRegistration:
        return DocumentRegistration(
            document_id=document_id,
            original_filename="README.md",
            file_extension=".md",
            file_size_bytes=None,
            upload_path=None,
            text_path=None,
            chunk_count=12,
            owner_id=self.OWNER_ID,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    def test_register_and_list_preserve_non_uuid_document_id(self):
        register_document(self.registration(), self.session_factory)

        documents = list_registered_documents(self.session_factory)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["document_id"], "eval-readme")
        self.assertEqual(documents[0]["filename"], "README.md")
        self.assertEqual(documents[0]["chunk_count"], 12)

    def test_list_filters_document_metadata_at_the_registry_query(self):
        register_document(self.registration("allowed-document"), self.session_factory)
        register_document(self.registration("blocked-document"), self.session_factory)

        documents = list_registered_documents(
            self.session_factory,
            document_ids={"allowed-document"},
        )

        self.assertEqual(
            [document["document_id"] for document in documents],
            ["allowed-document"],
        )
        self.assertEqual(
            list_registered_documents(self.session_factory, document_ids=set()),
            [],
        )

    def test_duplicate_registration_is_rejected(self):
        registration = self.registration()
        register_document(registration, self.session_factory)

        with self.assertRaises(DocumentRegistryConflictError):
            register_document(registration, self.session_factory)

    def test_backfill_is_idempotent(self):
        registration = self.registration()

        first = backfill_document_registry([registration], self.session_factory)
        second = backfill_document_registry([registration], self.session_factory)

        self.assertEqual((first.inserted, first.skipped_existing), (1, 0))
        self.assertEqual((second.inserted, second.skipped_existing), (0, 1))
        self.assertEqual(registered_document_ids(self.session_factory), {"eval-readme"})

    def test_backfill_rejects_conflicting_immutable_metadata(self):
        register_document(self.registration(), self.session_factory)
        conflicting = DocumentRegistration(
            **{
                **self.registration().__dict__,
                "original_filename": "different.md",
            }
        )

        with self.assertRaises(DocumentRegistryConflictError):
            backfill_document_registry([conflicting], self.session_factory)

    def test_database_failure_log_exposes_type_but_not_credentials(self):
        def broken_factory():
            raise OperationalError(
                "SELECT document_id FROM documents",
                {},
                RuntimeError("password=hunter2"),
            )

        with self.assertLogs(
            "app.services.document_registry",
            level="WARNING",
        ) as captured:
            with self.assertRaises(DocumentRegistryUnavailableError):
                verify_document_registry_available(broken_factory)

        log_text = "\n".join(captured.output)
        self.assertIn("error_type=OperationalError", log_text)
        self.assertNotIn("hunter2", log_text)
        self.assertNotIn("password", log_text)


class HistoricalRegistrationTests(unittest.TestCase):
    def test_builds_only_chunk_backed_documents_and_uses_relative_paths(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            upload_dir = root / "uploads"
            text_dir = root / "texts"
            index_dir = root / "index"
            upload_dir.mkdir()
            text_dir.mkdir()
            index_dir.mkdir()
            paths = DocumentStoragePaths(
                upload_dir=upload_dir,
                text_dir=text_dir,
                index_dir=index_dir,
                chunks_file=index_dir / "chunks.json",
                vectors_file=index_dir / "vectors.json",
            )
            upload_file = upload_dir / "legacy-id_policy.md"
            upload_file.write_bytes(b"policy")
            (text_dir / "legacy-id.txt").write_text("policy", encoding="utf-8")
            # This filesystem orphan has no chunks and must not become a row.
            (upload_dir / "orphan_unused.txt").write_bytes(b"unused")
            chunks = [
                {
                    "document_id": "legacy-id",
                    "filename": "policy.md",
                    "created_at": "2026-01-02T00:00:00+00:00",
                },
                {
                    "document_id": "legacy-id",
                    "filename": "policy.md",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
            ]

            owner_id = UUID("00000000-0000-0000-0000-000000000102")
            registrations = build_historical_registrations(
                chunks,
                paths,
                owner_id=owner_id,
            )

        self.assertEqual(len(registrations), 1)
        registration = registrations[0]
        self.assertEqual(registration.document_id, "legacy-id")
        self.assertEqual(registration.chunk_count, 2)
        self.assertEqual(registration.upload_path, "uploads/legacy-id_policy.md")
        self.assertEqual(registration.text_path, "texts/legacy-id.txt")
        self.assertEqual(registration.file_size_bytes, 6)
        self.assertEqual(registration.owner_id, owner_id)
        self.assertEqual(
            registration.created_at,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
