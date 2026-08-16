import unittest
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.services.user_registry import (
    USERNAME_MAX_LENGTH,
    InvalidPasswordHashError,
    InvalidUsernameError,
    UserRegistryUnavailableError,
    UsernameAlreadyExistsError,
    create_user,
    get_user_by_id,
    get_user_credential_by_username,
    get_user_by_username,
    normalize_username,
)


class UserRegistryTests(unittest.TestCase):
    HASH = "$argon2id$v=19$m=65536,t=3,p=4$synthetic$synthetic"

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

    def test_username_normalization_is_trimmed_and_case_insensitive(self):
        self.assertEqual(normalize_username("  Alice  "), "alice")
        self.assertEqual(normalize_username("ALICE"), "alice")

    def test_invalid_usernames_are_rejected(self):
        for username in ("", "   ", "x" * (USERNAME_MAX_LENGTH + 1)):
            with self.subTest(username=repr(username)):
                with self.assertRaises(InvalidUsernameError):
                    normalize_username(username)

    def test_create_and_lookup_preserve_stable_uuid(self):
        created = create_user(
            " Alice ",
            self.session_factory,
            password_hash=self.HASH,
        )

        by_id = get_user_by_id(str(created.user_id), self.session_factory)
        by_username = get_user_by_username("ALICE", self.session_factory)

        self.assertIsInstance(created.user_id, UUID)
        self.assertEqual(created.username, "alice")
        self.assertIsNotNone(created.created_at.tzinfo)
        self.assertIsNotNone(by_id)
        self.assertIsNotNone(by_username)
        self.assertEqual(by_id.user_id, created.user_id)
        self.assertEqual(by_username.user_id, created.user_id)

    def test_duplicate_canonical_username_is_rejected_and_next_write_works(self):
        create_user("Alice", self.session_factory, password_hash=self.HASH)

        with self.assertRaises(UsernameAlreadyExistsError):
            create_user(
                "  ALICE  ",
                self.session_factory,
                password_hash=self.HASH,
            )

        second = create_user("bob", self.session_factory, password_hash=self.HASH)
        self.assertEqual(second.username, "bob")

    def test_credential_lookup_is_separate_from_public_identity(self):
        created = create_user(
            "alice",
            self.session_factory,
            password_hash=self.HASH,
        )

        public = get_user_by_id(created.user_id, self.session_factory)
        credential = get_user_credential_by_username("ALICE", self.session_factory)

        self.assertFalse(hasattr(public, "password_hash"))
        self.assertIsNotNone(credential)
        self.assertEqual(credential.identity, public)
        self.assertEqual(credential.password_hash, self.HASH)

    def test_empty_password_hash_is_rejected_before_database_access(self):
        with self.assertRaises(InvalidPasswordHashError):
            create_user("alice", self.session_factory, password_hash="")

    def test_missing_users_return_none(self):
        self.assertIsNone(get_user_by_id(uuid4(), self.session_factory))
        self.assertIsNone(get_user_by_id("not-a-uuid", self.session_factory))
        self.assertIsNone(
            get_user_by_username("missing-user", self.session_factory)
        )

    def test_database_failure_log_exposes_type_but_not_credentials(self):
        def broken_factory():
            raise OperationalError(
                "INSERT INTO users",
                {},
                RuntimeError("password=hunter2"),
            )

        with self.assertLogs(
            "app.services.user_registry",
            level="WARNING",
        ) as captured:
            with self.assertRaises(UserRegistryUnavailableError):
                create_user(
                    "alice",
                    broken_factory,
                    password_hash=self.HASH,
                )

        log_text = "\n".join(captured.output)
        self.assertIn("error_type=OperationalError", log_text)
        self.assertNotIn("hunter2", log_text)
        self.assertNotIn("password", log_text)


if __name__ == "__main__":
    unittest.main()
