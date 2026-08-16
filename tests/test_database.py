import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import DateTime, String, Text, Uuid

from app.core.config import settings
from app.db.models import DocumentACLRecord, DocumentRecord, UserRecord
from app.db.session import (
    DatabaseConfigurationError,
    create_database_engine,
    require_test_database_url,
    session_scope,
)


class TestDatabaseUrlSafety(unittest.TestCase):
    def test_test_url_is_mandatory_and_never_falls_back(self):
        with patch.object(
            settings,
            "database_url",
            "postgresql+psycopg://user:secret@localhost/enterprise_rag",
        ), patch.object(settings, "test_database_url", ""):
            with self.assertRaises(DatabaseConfigurationError):
                require_test_database_url()

    def test_test_url_cannot_identify_the_production_database(self):
        with patch.object(
            settings,
            "database_url",
            "postgresql+psycopg://user:one@localhost/enterprise_rag",
        ), patch.object(
            settings,
            "test_database_url",
            "postgresql+psycopg://user:two@localhost/enterprise_rag",
        ):
            with self.assertRaises(DatabaseConfigurationError):
                require_test_database_url()

    def test_test_url_requires_an_explicit_test_database_name(self):
        with patch.object(settings, "database_url", ""), patch.object(
            settings,
            "test_database_url",
            "postgresql+psycopg://user:secret@localhost/not-production",
        ):
            with self.assertRaises(DatabaseConfigurationError):
                require_test_database_url()

    def test_distinct_test_url_is_returned(self):
        test_url = "postgresql+psycopg://user:secret@localhost/enterprise_rag_test"
        with patch.object(
            settings,
            "database_url",
            "postgresql+psycopg://user:secret@localhost/enterprise_rag",
        ), patch.object(settings, "test_database_url", test_url):
            self.assertEqual(require_test_database_url(), test_url)


class DatabaseInfrastructureTests(unittest.TestCase):
    def tearDown(self):
        create_database_engine.cache_clear()

    def test_engine_is_reused_for_the_same_url(self):
        first = create_database_engine("sqlite://")
        second = create_database_engine("sqlite://")

        self.assertIs(first, second)

    def test_session_scope_rolls_back_and_closes_on_failure(self):
        session = MagicMock()
        session_factory = MagicMock(return_value=session)

        with self.assertRaisesRegex(RuntimeError, "business failure"):
            with session_scope(session_factory):
                raise RuntimeError("business failure")

        session.commit.assert_not_called()
        session.rollback.assert_called_once_with()
        session.close.assert_called_once_with()

    def test_document_schema_uses_stable_string_primary_key(self):
        table = DocumentRecord.__table__

        self.assertEqual(
            set(table.columns.keys()),
            {
                "document_id",
                "original_filename",
                "file_extension",
                "file_size_bytes",
                "upload_path",
                "text_path",
                "chunk_count",
                "owner_id",
                "created_at",
            },
        )
        self.assertTrue(table.c.document_id.primary_key)
        self.assertIsInstance(table.c.document_id.type, String)
        self.assertIsInstance(table.c.owner_id.type, Uuid)
        self.assertFalse(table.c.owner_id.nullable)
        self.assertFalse(
            {"user_id", "created_by", "uploaded_by"} & set(table.columns.keys())
        )
        owner_foreign_keys = {
            (
                foreign_key.parent.name,
                foreign_key.target_fullname,
                foreign_key.ondelete,
            )
            for foreign_key in table.foreign_keys
        }
        self.assertIn(
            ("owner_id", "users.user_id", "RESTRICT"),
            owner_foreign_keys,
        )

    def test_user_schema_has_uuid_identity_and_one_way_credential(self):
        table = UserRecord.__table__

        self.assertEqual(
            set(table.columns.keys()),
            {"user_id", "username", "password_hash", "created_at"},
        )
        self.assertTrue(table.c.user_id.primary_key)
        self.assertIsInstance(table.c.user_id.type, Uuid)
        self.assertIsInstance(table.c.created_at.type, DateTime)
        self.assertTrue(table.c.created_at.type.timezone)
        self.assertFalse(table.c.created_at.nullable)
        self.assertIsInstance(table.c.password_hash.type, Text)
        self.assertFalse(table.c.password_hash.nullable)
        self.assertNotIn("password", table.columns)

        unique_column_sets = {
            tuple(constraint.columns.keys())
            for constraint in table.constraints
            if constraint.__class__.__name__ == "UniqueConstraint"
        }
        self.assertIn(("username",), unique_column_sets)

    def test_acl_schema_uses_stable_composite_identity(self):
        table = DocumentACLRecord.__table__

        self.assertEqual(
            set(table.columns.keys()),
            {"document_id", "user_id", "created_at"},
        )
        self.assertEqual(
            tuple(column.name for column in table.primary_key.columns),
            ("document_id", "user_id"),
        )
        self.assertIsInstance(table.c.document_id.type, String)
        self.assertIsInstance(table.c.user_id.type, Uuid)
        self.assertIsInstance(table.c.created_at.type, DateTime)
        self.assertTrue(table.c.created_at.type.timezone)
        foreign_keys = {
            (
                foreign_key.parent.name,
                foreign_key.target_fullname,
                foreign_key.ondelete,
            )
            for foreign_key in table.foreign_keys
        }
        self.assertEqual(
            foreign_keys,
            {
                ("document_id", "documents.document_id", "CASCADE"),
                ("user_id", "users.user_id", "CASCADE"),
            },
        )


if __name__ == "__main__":
    unittest.main()
