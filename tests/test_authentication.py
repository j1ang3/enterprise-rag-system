from datetime import datetime, timedelta, timezone
from io import StringIO
import logging
import unittest
from unittest.mock import patch
from uuid import uuid4

import jwt
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.passwords import (
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    PasswordValidationError,
    hash_password,
    verify_password,
)
from app.auth.tokens import (
    JWT_ALGORITHM,
    AuthenticationConfigurationError,
    InvalidAccessTokenError,
    create_access_token,
    decode_access_token,
    require_jwt_secret,
)
from app.core.config import settings
from app.db.base import Base
from app.db.models import UserRecord
from app.main import app


TEST_SECRET = "w10-t3-test-only-jwt-secret-with-more-than-64-bytes-of-entropy-material"


def _tamper_signature(token: str) -> str:
    header, payload, signature = token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    return ".".join((header, payload, replacement + signature[1:]))


class PasswordHashingTests(unittest.TestCase):
    def test_argon2id_hashes_are_salted_and_verify(self):
        password = "correct horse battery staple"
        first = hash_password(password)
        second = hash_password(password)

        self.assertTrue(first.startswith("$argon2id$"))
        self.assertNotEqual(first, second)
        self.assertTrue(verify_password(password, first))
        self.assertFalse(verify_password("wrong password", first))

    def test_password_policy_rejects_short_and_long_values(self):
        for password in (
            "",
            "x" * (PASSWORD_MIN_LENGTH - 1),
            "x" * (PASSWORD_MAX_LENGTH + 1),
        ):
            with self.subTest(length=len(password)):
                with self.assertRaises(PasswordValidationError):
                    hash_password(password)

    def test_invalid_encoded_hash_fails_closed(self):
        self.assertFalse(verify_password("valid password", "not-an-argon2-hash"))


class AccessTokenTests(unittest.TestCase):
    def setUp(self):
        self.secret_patch = patch.object(
            settings,
            "jwt_secret_key",
            SecretStr(TEST_SECRET),
        )
        self.secret_patch.start()

    def tearDown(self):
        self.secret_patch.stop()

    def test_valid_token_has_only_required_identity_claims(self):
        user_id = uuid4()
        token = create_access_token(user_id)
        payload = jwt.decode(
            token.value,
            TEST_SECRET,
            algorithms=[JWT_ALGORITHM],
        )

        self.assertEqual(decode_access_token(token.value), user_id)
        self.assertEqual(set(payload), {"sub", "iat", "exp"})
        self.assertEqual(payload["sub"], str(user_id))
        self.assertNotIn(TEST_SECRET, token.value)

    def test_expired_modified_malformed_and_unsupported_tokens_are_rejected(self):
        user_id = uuid4()
        past = datetime.now(timezone.utc) - timedelta(minutes=2)
        expired = jwt.encode(
            {
                "sub": str(user_id),
                "iat": past,
                "exp": past + timedelta(minutes=1),
            },
            TEST_SECRET,
            algorithm=JWT_ALGORITHM,
        )
        unsupported = jwt.encode(
            {
                "sub": str(user_id),
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=1),
            },
            TEST_SECRET,
            algorithm="HS384",
        )
        unsigned = jwt.encode(
            {
                "sub": str(user_id),
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=1),
            },
            key="",
            algorithm="none",
        )
        valid = create_access_token(user_id).value

        for token in (
            expired,
            unsupported,
            unsigned,
            "malformed",
            _tamper_signature(valid),
        ):
            with self.subTest(token_kind=token.count(".")):
                with self.assertRaises(InvalidAccessTokenError):
                    decode_access_token(token)

    def test_missing_sub_and_invalid_user_id_are_rejected(self):
        now = datetime.now(timezone.utc)
        for subject_payload in ({}, {"sub": "not-a-uuid"}):
            token = jwt.encode(
                {**subject_payload, "iat": now, "exp": now + timedelta(minutes=1)},
                TEST_SECRET,
                algorithm=JWT_ALGORITHM,
            )
            with self.assertRaises(InvalidAccessTokenError):
                decode_access_token(token)

    def test_missing_or_short_secret_fails_closed_without_disclosure(self):
        for secret in ("", "too-short"):
            with patch.object(settings, "jwt_secret_key", SecretStr(secret)):
                with self.assertRaises(AuthenticationConfigurationError) as captured:
                    require_jwt_secret()
                self.assertNotIn(secret or "unused", str(captured.exception))


class AuthenticationApiTests(unittest.TestCase):
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
        self.session_patch = patch(
            "app.services.user_registry.get_session_factory",
            return_value=self.session_factory,
        )
        self.secret_patch = patch.object(
            settings,
            "jwt_secret_key",
            SecretStr(TEST_SECRET),
        )
        self.session_patch.start()
        self.secret_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.secret_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()

    def _register(self, username: str = " Alice ", password: str = "safe password"):
        return self.client.post(
            "/auth/register",
            json={"username": username, "password": password},
        )

    def _login(self, username: str = "ALICE", password: str = "safe password"):
        return self.client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )

    def test_register_login_and_me_round_trip(self):
        plaintext = "safe password"
        registered = self._register(password=plaintext)
        self.assertEqual(registered.status_code, 201)
        registered_data = registered.json()["data"]
        self.assertEqual(registered_data["username"], "alice")
        self.assertNotIn("password", registered.text.lower())
        self.assertNotIn("hash", registered.text.lower())

        with self.session_factory() as session:
            record = session.scalar(
                select(UserRecord).where(UserRecord.username == "alice")
            )
        self.assertIsNotNone(record)
        self.assertNotEqual(record.password_hash, plaintext)
        self.assertNotIn(plaintext, record.password_hash)
        self.assertTrue(verify_password(plaintext, record.password_hash))

        logged_in = self._login(password=plaintext)
        self.assertEqual(logged_in.status_code, 200)
        token_data = logged_in.json()["data"]
        self.assertEqual(token_data["token_type"], "bearer")
        self.assertEqual(token_data["expires_in"], 1800)
        self.assertNotIn(TEST_SECRET, logged_in.text)

        current = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["data"]["user_id"], registered_data["user_id"])
        self.assertEqual(current.json()["data"]["username"], "alice")

    def test_duplicate_registration_is_conflict(self):
        self.assertEqual(self._register().status_code, 201)
        duplicate = self._register(username="  ALICE  ")
        self.assertEqual(duplicate.status_code, 409)

    def test_registration_password_boundaries_are_validation_errors(self):
        for password in ("", "short", "x" * (PASSWORD_MAX_LENGTH + 1)):
            with self.subTest(length=len(password)):
                response = self._register(password=password)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(password or "unused-empty-value", response.text)

    def test_wrong_password_and_unknown_user_have_same_public_error(self):
        self.assertEqual(self._register().status_code, 201)
        wrong = self._login(password="wrong password")
        unknown = self._login(username="unknown", password="wrong password")

        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())
        self.assertEqual(wrong.headers["www-authenticate"], "Bearer")

    def test_bearer_errors_and_deleted_user_are_unauthorized(self):
        self.assertEqual(self._register().status_code, 201)
        token = self._login().json()["data"]["access_token"]

        for headers in ({}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer malformed"}):
            response = self.client.get("/auth/me", headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers["www-authenticate"], "Bearer")

        user_id = decode_access_token(token)
        with self.session_factory() as session:
            with session.begin():
                session.execute(
                    delete(UserRecord).where(UserRecord.user_id == user_id)
                )
        missing = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(missing.status_code, 401)

    def test_missing_secret_blocks_registration_without_creating_user(self):
        with patch.object(settings, "jwt_secret_key", SecretStr("")):
            response = self._register()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(TEST_SECRET, response.text)
        with self.session_factory() as session:
            self.assertIsNone(session.scalar(select(UserRecord)))

    def test_password_and_secret_are_not_logged_on_authentication_failure(self):
        plaintext = "unique password value for log test"
        self.assertEqual(self._register(password=plaintext).status_code, 201)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            response = self._login(password="different wrong password")
        finally:
            root_logger.removeHandler(handler)
        self.assertEqual(response.status_code, 401)
        emitted = stream.getvalue()
        self.assertNotIn(plaintext, emitted)
        self.assertNotIn(TEST_SECRET, emitted)


if __name__ == "__main__":
    unittest.main()
