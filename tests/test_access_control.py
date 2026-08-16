from datetime import datetime, timezone
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.dependencies import get_current_user
from app.db.base import Base
from app.db.models import DocumentACLRecord
from app.main import app
from app.services.access_control import (
    ACLManagementForbiddenError,
    DocumentGrant,
    DocumentNotFoundError,
    DuplicateGrantError,
    GrantNotFoundError,
    OwnerSelfShareError,
    TargetUserNotFoundError,
    can_user_read_document,
    get_readable_document_ids,
    grant_document_read_access,
    list_document_read_grants,
    revoke_document_read_access,
)
from app.services.document_registry import DocumentRegistration, register_document
from app.services.user_registry import UserIdentity, create_user


ENCODED_TEST_HASH = "$argon2id$test-only-encoded-value"


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


class AccessControlServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = _sqlite_engine()
        Base.metadata.create_all(self.engine)
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
        self.carol = create_user(
            "carol",
            self.session_factory,
            password_hash=ENCODED_TEST_HASH,
        )
        register_document(
            DocumentRegistration(
                document_id="document-a",
                original_filename="document-a.txt",
                file_extension=".txt",
                file_size_bytes=8,
                upload_path=None,
                text_path=None,
                chunk_count=1,
                owner_id=self.alice.user_id,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            self.session_factory,
        )

    def tearDown(self):
        self.engine.dispose()

    def test_default_deny_owner_allow_grant_and_immediate_revoke(self):
        self.assertTrue(
            can_user_read_document(
                self.alice.user_id,
                "document-a",
                self.session_factory,
            )
        )
        self.assertFalse(
            can_user_read_document(
                self.bob.user_id,
                "document-a",
                self.session_factory,
            )
        )
        self.assertFalse(
            can_user_read_document(
                self.carol.user_id,
                "document-a",
                self.session_factory,
            )
        )

        grant_document_read_access(
            "document-a",
            self.bob.user_id,
            self.alice.user_id,
            self.session_factory,
        )
        self.assertTrue(
            can_user_read_document(
                self.bob.user_id,
                "document-a",
                self.session_factory,
            )
        )
        self.assertFalse(
            can_user_read_document(
                self.carol.user_id,
                "document-a",
                self.session_factory,
            )
        )

        revoke_document_read_access(
            "document-a",
            self.bob.user_id,
            self.alice.user_id,
            self.session_factory,
        )
        self.assertFalse(
            can_user_read_document(
                self.bob.user_id,
                "document-a",
                self.session_factory,
            )
        )

    def test_bulk_readable_documents_union_is_deduplicated_and_one_query(self):
        for document_id, owner in (
            ("document-b", self.bob),
            ("document-c", self.carol),
        ):
            register_document(
                DocumentRegistration(
                    document_id=document_id,
                    original_filename=f"{document_id}.txt",
                    file_extension=".txt",
                    file_size_bytes=8,
                    upload_path=None,
                    text_path=None,
                    chunk_count=1,
                    owner_id=owner.user_id,
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
                self.session_factory,
            )
        grant_document_read_access(
            "document-a",
            self.bob.user_id,
            self.alice.user_id,
            self.session_factory,
        )
        # Legacy/anomalous owner ACL data must not duplicate the UNION result.
        with self.session_factory() as session, session.begin():
            session.add(
                DocumentACLRecord(
                    document_id="document-a",
                    user_id=self.alice.user_id,
                    created_at=datetime.now(timezone.utc),
                )
            )

        select_count = 0

        def count_selects(_conn, _cursor, statement, _parameters, _context, _executemany):
            nonlocal select_count
            if statement.lstrip().upper().startswith("SELECT"):
                select_count += 1

        event.listen(self.engine, "before_cursor_execute", count_selects)
        try:
            readable = get_readable_document_ids(
                self.alice.user_id,
                self.session_factory,
            )
        finally:
            event.remove(self.engine, "before_cursor_execute", count_selects)

        self.assertEqual(readable, frozenset({"document-a"}))
        self.assertEqual(select_count, 1)
        self.assertEqual(
            get_readable_document_ids(self.bob.user_id, self.session_factory),
            frozenset({"document-a", "document-b"}),
        )
        self.assertEqual(
            get_readable_document_ids(self.carol.user_id, self.session_factory),
            frozenset({"document-c"}),
        )

    def test_owner_has_no_acl_row_and_cannot_self_share(self):
        with self.session_factory() as session:
            self.assertIsNone(
                session.get(
                    DocumentACLRecord,
                    ("document-a", self.alice.user_id),
                )
            )
        with self.assertRaises(OwnerSelfShareError):
            grant_document_read_access(
                "document-a",
                self.alice.user_id,
                self.alice.user_id,
                self.session_factory,
            )

    def test_duplicate_missing_and_invalid_targets_have_explicit_semantics(self):
        grant_document_read_access(
            "document-a",
            self.bob.user_id,
            self.alice.user_id,
            self.session_factory,
        )
        with self.assertRaises(DuplicateGrantError):
            grant_document_read_access(
                "document-a",
                self.bob.user_id,
                self.alice.user_id,
                self.session_factory,
            )
        with self.assertRaises(GrantNotFoundError):
            revoke_document_read_access(
                "document-a",
                self.carol.user_id,
                self.alice.user_id,
                self.session_factory,
            )
        with self.assertRaises(TargetUserNotFoundError):
            grant_document_read_access(
                "document-a",
                uuid4(),
                self.alice.user_id,
                self.session_factory,
            )
        with self.assertRaises(DocumentNotFoundError):
            can_user_read_document(
                self.alice.user_id,
                "missing",
                self.session_factory,
            )

    def test_non_owner_cannot_grant_revoke_or_list(self):
        grant_document_read_access(
            "document-a",
            self.bob.user_id,
            self.alice.user_id,
            self.session_factory,
        )
        operations = (
            lambda: grant_document_read_access(
                "document-a",
                self.carol.user_id,
                self.bob.user_id,
                self.session_factory,
            ),
            lambda: revoke_document_read_access(
                "document-a",
                self.carol.user_id,
                self.bob.user_id,
                self.session_factory,
            ),
            lambda: list_document_read_grants(
                "document-a",
                self.bob.user_id,
                self.session_factory,
            ),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ACLManagementForbiddenError):
                    operation()

    def test_owner_lists_only_safe_recipient_identity(self):
        grant = grant_document_read_access(
            "document-a",
            self.bob.user_id,
            self.alice.user_id,
            self.session_factory,
        )
        listed = list_document_read_grants(
            "document-a",
            self.alice.user_id,
            self.session_factory,
        )

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].document_id, grant.document_id)
        self.assertEqual(listed[0].user_id, grant.user_id)
        self.assertEqual(listed[0].username, grant.username)
        self.assertFalse(hasattr(listed[0], "password_hash"))

    def test_database_enforces_composite_uniqueness_and_foreign_keys(self):
        created_at = datetime.now(timezone.utc)
        with self.session_factory() as session:
            with session.begin():
                session.add(
                    DocumentACLRecord(
                        document_id="document-a",
                        user_id=self.bob.user_id,
                        created_at=created_at,
                    )
                )
        invalid_rows = (
            DocumentACLRecord(
                document_id="document-a",
                user_id=self.bob.user_id,
                created_at=created_at,
            ),
            DocumentACLRecord(
                document_id="missing",
                user_id=self.carol.user_id,
                created_at=created_at,
            ),
            DocumentACLRecord(
                document_id="document-a",
                user_id=uuid4(),
                created_at=created_at,
            ),
        )
        for row in invalid_rows:
            with self.subTest(document_id=row.document_id, user_id=row.user_id):
                with self.assertRaises(IntegrityError):
                    with self.session_factory() as session:
                        with session.begin():
                            session.add(row)


class AccessControlApiTests(unittest.TestCase):
    def setUp(self):
        self.owner = UserIdentity(
            user_id=uuid4(),
            username="owner",
            created_at=datetime.now(timezone.utc),
        )
        self.target_id = uuid4()
        app.dependency_overrides[get_current_user] = lambda: self.owner
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_current_user, None)

    def test_grant_list_and_revoke_api_contract(self):
        grant = DocumentGrant(
            document_id="document-a",
            user_id=self.target_id,
            username="recipient",
            created_at=datetime.now(timezone.utc),
        )
        with patch(
            "app.routers.documents.grant_document_read_access",
            return_value=grant,
        ), patch(
            "app.routers.documents.list_document_read_grants",
            return_value=[grant],
        ), patch(
            "app.routers.documents.revoke_document_read_access",
        ):
            granted = self.client.post(
                "/documents/document-a/shares",
                json={"user_id": str(self.target_id)},
            )
            listed = self.client.get("/documents/document-a/shares")
            revoked = self.client.delete(
                f"/documents/document-a/shares/{self.target_id}"
            )

        self.assertEqual(granted.status_code, 201)
        self.assertEqual(granted.json()["data"]["username"], "recipient")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["data"]["shares"]), 1)
        self.assertEqual(revoked.status_code, 200)

    def test_management_errors_map_to_safe_http_statuses(self):
        cases = (
            (DocumentNotFoundError("Document does not exist."), 404),
            (TargetUserNotFoundError("Share recipient does not exist."), 404),
            (ACLManagementForbiddenError("forbidden"), 403),
            (DuplicateGrantError("duplicate"), 409),
            (OwnerSelfShareError("self"), 409),
        )
        for error, expected_status in cases:
            with self.subTest(error=type(error).__name__), patch(
                "app.routers.documents.grant_document_read_access",
                side_effect=error,
            ):
                response = self.client.post(
                    "/documents/document-a/shares",
                    json={"user_id": str(self.target_id)},
                )
            self.assertEqual(response.status_code, expected_status)

    def test_all_management_endpoints_require_authentication(self):
        app.dependency_overrides.pop(get_current_user, None)
        responses = (
            self.client.post(
                "/documents/document-a/shares",
                json={"user_id": str(self.target_id)},
            ),
            self.client.get("/documents/document-a/shares"),
            self.client.delete(
                f"/documents/document-a/shares/{self.target_id}"
            ),
        )
        self.assertEqual([response.status_code for response in responses], [401] * 3)


if __name__ == "__main__":
    unittest.main()
