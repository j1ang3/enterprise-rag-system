import sys
from datetime import datetime, timedelta, timezone
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

from app.auth.passwords import verify_password
from app.auth.tokens import JWT_ALGORITHM
from app.core.config import settings
from app.db.models import DocumentRecord, UserRecord
from app.db.session import (
    create_database_engine,
    create_session_factory,
    require_test_database_url,
)
from app.main import app


TEST_JWT_SECRET = (
    "w10-t3-real-postgresql-test-only-jwt-secret-with-more-than-64-bytes"
)
EXPECTED_USER_COLUMNS = {"user_id", "username", "password_hash", "created_at"}


def _upgrade_test_database(database_url: str) -> Config:
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.attributes["database_url"] = database_url
    command.upgrade(alembic_config, "head")
    return alembic_config


def _tamper_signature(token: str) -> str:
    header, payload, signature = token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    return ".".join((header, payload, replacement + signature[1:]))


def main() -> int:
    # This gate rejects missing, production-identical, and non-_test URLs.
    test_database_url = require_test_database_url()
    engine = create_database_engine(test_database_url)
    session_factory = create_session_factory(test_database_url)

    inspector = inspect(engine)
    document_columns_before = {
        column["name"] for column in inspector.get_columns("documents")
    }
    with engine.connect() as connection:
        document_ids_before = set(
            connection.execute(select(DocumentRecord.document_id)).scalars()
        )
    with session_factory() as session:
        preexisting_user_ids = set(session.scalars(select(UserRecord.user_id)).all())

    alembic_config = _upgrade_test_database(test_database_url)
    inspector = inspect(engine)
    user_columns = {
        column["name"] for column in inspector.get_columns("users")
    }
    if user_columns != EXPECTED_USER_COLUMNS:
        raise RuntimeError("The test users table does not match the W10-T3 schema.")
    columns_by_name = {
        column["name"]: column for column in inspector.get_columns("users")
    }
    if columns_by_name["password_hash"]["nullable"]:
        raise RuntimeError("password_hash must be NOT NULL.")

    unique_constraints = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("users")
    }
    if ("username",) not in unique_constraints:
        raise RuntimeError("PostgreSQL no longer enforces unique usernames.")

    document_columns_after = {
        column["name"] for column in inspector.get_columns("documents")
    }
    if document_columns_after != document_columns_before:
        raise RuntimeError("W10-T3 changed the documents schema.")
    synthetic_username = f" W10T3.User.{uuid4()} "
    synthetic_password = "W10-T3 real PostgreSQL password"
    created_user_id: UUID | None = None
    try:
        with patch.object(settings, "database_url", test_database_url), patch.object(
            settings,
            "jwt_secret_key",
            SecretStr(TEST_JWT_SECRET),
        ):
            client = TestClient(app)
            registered = client.post(
                "/auth/register",
                json={
                    "username": synthetic_username,
                    "password": synthetic_password,
                },
            )
            if registered.status_code != 201:
                raise RuntimeError("Real PostgreSQL registration failed.")
            registered_body = registered.json()
            created_user_id = UUID(registered_body["data"]["user_id"])
            if "password" in registered.text.lower() or "hash" in registered.text.lower():
                raise RuntimeError("Registration response exposed credential fields.")

            with session_factory() as session:
                record = session.get(UserRecord, created_user_id)
            if record is None:
                raise RuntimeError("Registered user was not committed to PostgreSQL.")
            if record.password_hash == synthetic_password:
                raise RuntimeError("PostgreSQL stored a plaintext password.")
            if synthetic_password in record.password_hash:
                raise RuntimeError("Encoded credential contains the plaintext password.")
            if not record.password_hash.startswith("$argon2id$"):
                raise RuntimeError("PostgreSQL credential is not an Argon2id hash.")
            if not verify_password(synthetic_password, record.password_hash):
                raise RuntimeError("Persisted hash did not verify the submitted password.")

            duplicate = client.post(
                "/auth/register",
                json={
                    "username": synthetic_username.upper(),
                    "password": synthetic_password,
                },
            )
            if duplicate.status_code != 409:
                raise RuntimeError("Duplicate canonical registration was not rejected.")

            wrong_password = client.post(
                "/auth/login",
                json={
                    "username": synthetic_username,
                    "password": "W10-T3 incorrect password",
                },
            )
            unknown_user = client.post(
                "/auth/login",
                json={
                    "username": f"missing-{uuid4()}",
                    "password": "W10-T3 incorrect password",
                },
            )
            if wrong_password.status_code != 401 or unknown_user.status_code != 401:
                raise RuntimeError("Invalid credentials were not rejected.")
            if wrong_password.json() != unknown_user.json():
                raise RuntimeError("Login response reveals whether a username exists.")

            logged_in = client.post(
                "/auth/login",
                json={
                    "username": synthetic_username.upper(),
                    "password": synthetic_password,
                },
            )
            if logged_in.status_code != 200:
                raise RuntimeError("Real PostgreSQL login failed.")
            access_token = logged_in.json()["data"]["access_token"]
            payload = jwt.decode(
                access_token,
                TEST_JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
            )
            if set(payload) != {"sub", "iat", "exp"}:
                raise RuntimeError("JWT contains claims outside the W10-T3 scope.")
            if payload["sub"] != str(created_user_id):
                raise RuntimeError("JWT sub is not the stable user_id.")

            current = client.get(
                "/auth/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if current.status_code != 200:
                raise RuntimeError("Bearer JWT did not resolve /auth/me.")
            if UUID(current.json()["data"]["user_id"]) != created_user_id:
                raise RuntimeError("/auth/me resolved a different user.")

            tampered = client.get(
                "/auth/me",
                headers={"Authorization": f"Bearer {_tamper_signature(access_token)}"},
            )
            if tampered.status_code != 401:
                raise RuntimeError("Tampered JWT was accepted.")

            past = datetime.now(timezone.utc) - timedelta(minutes=2)
            expired_token = jwt.encode(
                {
                    "sub": str(created_user_id),
                    "iat": past,
                    "exp": past + timedelta(minutes=1),
                },
                TEST_JWT_SECRET,
                algorithm=JWT_ALGORITHM,
            )
            expired = client.get(
                "/auth/me",
                headers={"Authorization": f"Bearer {expired_token}"},
            )
            if expired.status_code != 401:
                raise RuntimeError("Expired JWT was accepted.")

            with session_factory() as session:
                with session.begin():
                    session.execute(
                        delete(UserRecord).where(
                            UserRecord.user_id == created_user_id
                        )
                    )
            missing_user = client.get(
                "/auth/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if missing_user.status_code != 401:
                raise RuntimeError("A deleted user still resolved from JWT payload alone.")
    finally:
        if created_user_id is not None:
            with session_factory() as session:
                with session.begin():
                    session.execute(
                        delete(UserRecord).where(
                            UserRecord.user_id == created_user_id
                        )
                    )

    with session_factory() as session:
        final_user_ids = set(session.scalars(select(UserRecord.user_id)).all())
    if final_user_ids != preexisting_user_ids:
        raise RuntimeError("Authentication verification left User residue behind.")

    with engine.connect() as connection:
        current_revision = MigrationContext.configure(connection).get_current_revision()
        document_ids_after = set(
            connection.execute(select(DocumentRecord.document_id)).scalars()
        )
    if current_revision not in set(
        ScriptDirectory.from_config(alembic_config).get_heads()
    ):
        raise RuntimeError("Test database is not at the Alembic head revision.")
    if document_ids_after != document_ids_before:
        raise RuntimeError("Authentication verification changed document identities.")

    print(
        "Real PostgreSQL authentication verification passed: "
        "database=enterprise_rag_test, "
        f"revision={current_revision}, user_columns={len(user_columns)}, "
        "registration=true, argon2id_hash=true, login=true, jwt=true, "
        "current_user_lookup=true, wrong_password_rejected=true, "
        "tampered_rejected=true, expired_rejected=true, missing_user_rejected=true, "
        "documents_unchanged=true, exact_cleanup=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
