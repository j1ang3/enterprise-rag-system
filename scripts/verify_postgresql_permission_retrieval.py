"""Verify W10-T6 against the isolated real PostgreSQL test database."""

from __future__ import annotations

import json
import sys
import tempfile
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
from pydantic import SecretStr
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from app.auth.tokens import create_access_token
from app.core.config import settings
from app.db.models import DocumentACLRecord, DocumentRecord, UserRecord
from app.db.session import (
    create_database_engine,
    create_session_factory,
    require_test_database_url,
)
from app.main import app
from app.services import vector_store
from app.services.access_control import (
    get_readable_document_ids,
    grant_document_read_access,
    revoke_document_read_access,
)
from app.services.storage_paths import DocumentStoragePaths


TEST_JWT_SECRET = (
    "w10-t6-real-postgresql-test-only-jwt-secret-with-more-than-64-bytes"
)
TEST_EMBEDDING_MODEL = "w10-t6-controlled-embedding"
UNAUTHORIZED_SENTINEL = "W10T6_UNAUTHORIZED_SENTINEL"
ENCODED_TEST_HASH = "$argon2id$w10-t6-test-only-encoded-value"


def _upgrade_test_database(database_url: str) -> tuple[Config, str | None]:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")
    engine = create_database_engine(database_url)
    with engine.connect() as connection:
        revision = MigrationContext.configure(connection).get_current_revision()
    if revision not in set(ScriptDirectory.from_config(config).get_heads()):
        raise RuntimeError("TEST_DATABASE_URL is not at the Alembic head.")
    return config, revision


def _document(document_id: str, owner_id: UUID) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        file_extension=".txt",
        file_size_bytes=32,
        upload_path=None,
        text_path=None,
        chunk_count=1,
        owner_id=owner_id,
        created_at=datetime.now(timezone.utc),
    )


def _chunk(document_id: str, content: str) -> dict[str, object]:
    return {
        "chunk_id": f"{document_id}-1",
        "document_id": document_id,
        "filename": f"{document_id}.txt",
        "position": 1,
        "chunk_index": 0,
        "page_number": None,
        "content": content,
        "token_count": len(content.split()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _vector_entry(chunk: dict[str, object], score: float) -> dict[str, object]:
    return {
        **chunk,
        "embedding": [score, 0.0],
        "embedding_model": TEST_EMBEDDING_MODEL,
    }


def _snapshot(session_factory) -> tuple[set[UUID], set[str], set[tuple[str, UUID]]]:
    with session_factory() as session:
        return (
            set(session.scalars(select(UserRecord.user_id)).all()),
            set(session.scalars(select(DocumentRecord.document_id)).all()),
            set(
                session.execute(
                    select(DocumentACLRecord.document_id, DocumentACLRecord.user_id)
                ).all()
            ),
        )


def _bearer(user_id: UUID) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id).value}"}


def main() -> int:
    test_database_url = require_test_database_url()
    _, revision = _upgrade_test_database(test_database_url)
    session_factory = create_session_factory(test_database_url)
    before = _snapshot(session_factory)

    suffix = uuid4().hex
    user_ids = {name: uuid4() for name in ("alice", "bob", "carol")}
    document_ids = {
        name: f"w10t6-{name}-{suffix}" for name in ("a", "b", "c")
    }
    created_at = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory(prefix="enterprise-rag-w10t6-") as temp_dir:
        root = Path(temp_dir)
        paths = DocumentStoragePaths(
            upload_dir=root / "uploads",
            text_dir=root / "texts",
            index_dir=root / "index",
            chunks_file=root / "index" / "chunks.json",
            vectors_file=root / "index" / "vectors.json",
        )
        chunks = [
            _chunk(document_ids["a"], "Alice shared policy access fact."),
            _chunk(document_ids["b"], "Bob owns the zanzibar itinerary."),
            _chunk(
                document_ids["c"],
                f"Carol confidential policy {UNAUTHORIZED_SENTINEL}.",
            ),
        ]
        missing_identity = {
            key: value
            for key, value in _chunk("missing", "unknown high rank").items()
            if key != "document_id"
        }
        paths.index_dir.mkdir(parents=True, exist_ok=True)
        paths.chunks_file.write_text(json.dumps(chunks), encoding="utf-8")
        vector_store._save_vectors(
            [
                _vector_entry(chunks[2], 0.99),
                {**_vector_entry(missing_identity, 0.98)},
                _vector_entry(chunks[0], 0.70),
                _vector_entry(chunks[1], 0.60),
            ],
            paths.vectors_file,
        )

        try:
            with session_factory() as session, session.begin():
                for name, user_id in user_ids.items():
                    session.add(
                        UserRecord(
                            user_id=user_id,
                            username=f"w10t6-{name}-{suffix}",
                            password_hash=ENCODED_TEST_HASH,
                            created_at=created_at,
                        )
                    )
                session.flush()
                session.add_all(
                    [
                        _document(document_ids["a"], user_ids["alice"]),
                        _document(document_ids["b"], user_ids["bob"]),
                        _document(document_ids["c"], user_ids["carol"]),
                    ]
                )

            grant_document_read_access(
                document_ids["a"],
                user_ids["bob"],
                user_ids["alice"],
                session_factory,
            )
            expected = {
                "alice": frozenset({document_ids["a"]}),
                "bob": frozenset({document_ids["a"], document_ids["b"]}),
                "carol": frozenset({document_ids["c"]}),
            }
            for name, user_id in user_ids.items():
                actual = get_readable_document_ids(user_id, session_factory)
                if actual != expected[name]:
                    raise RuntimeError(f"PostgreSQL readable-document UNION failed for {name}.")

            captured_prompts: list[str] = []

            def fake_completion(messages, **_kwargs):
                captured_prompts.append(json.dumps(messages, ensure_ascii=False))
                return "Controlled verification answer."

            with patch.object(settings, "database_url", test_database_url), patch.object(
                settings,
                "jwt_secret_key",
                SecretStr(TEST_JWT_SECRET),
            ), patch.object(
                settings,
                "vector_store_backend",
                "faiss",
            ), patch(
                "app.services.knowledge_base.get_document_storage_paths",
                return_value=paths,
            ), patch(
                "app.services.vector_store.get_document_storage_paths",
                return_value=paths,
            ), patch(
                "app.services.vector_store.embed_text",
                return_value=[1.0, 0.0],
            ), patch(
                "app.services.vector_store.embedding_model_for_vector",
                return_value=TEST_EMBEDDING_MODEL,
            ), patch(
                "app.services.knowledge_base.is_llm_configured",
                return_value=True,
            ), patch(
                "app.services.knowledge_base.chat_completion",
                side_effect=fake_completion,
            ):
                client = TestClient(app)
                missing_auth = client.post("/search", json={"query": "policy"})
                invalid_auth = client.post(
                    "/chat/",
                    headers={"Authorization": "Bearer invalid-token"},
                    json={"question": "policy"},
                )
                if [missing_auth.status_code, invalid_auth.status_code] != [401, 401]:
                    raise RuntimeError("Authenticated retrieval entry points accepted bad auth.")

                bob_headers = _bearer(user_ids["bob"])
                bob_vector = client.post(
                    "/search",
                    headers=bob_headers,
                    json={"query": "policy", "top_k": 2, "min_score": 0.0},
                )
                if bob_vector.status_code != 200:
                    raise RuntimeError("Bob vector retrieval failed.")
                bob_vector_ids = {
                    row["document_id"]
                    for row in bob_vector.json()["data"]["results"]
                }
                if bob_vector_ids != expected["bob"]:
                    raise RuntimeError("FAISS did not recover exactly Bob's authorized top_k.")

                alice_chat = client.post(
                    "/chat/",
                    headers=_bearer(user_ids["alice"]),
                    json={
                        "question": "Alice shared policy access fact",
                        "retrieval_mode": "keyword",
                        "top_k": 2,
                        "min_score": 0.0,
                    },
                )
                bob_shared = client.post(
                    "/chat/",
                    headers=bob_headers,
                    json={
                        "question": "Alice shared policy access fact",
                        "retrieval_mode": "keyword",
                        "top_k": 2,
                        "min_score": 0.0,
                    },
                )
                for response, label in ((alice_chat, "owner"), (bob_shared, "ACL shared")):
                    if response.status_code != 200:
                        raise RuntimeError(f"{label} RAG retrieval failed.")
                    data = response.json()["data"]
                    if {row["document_id"] for row in data["contexts"]} != {
                        document_ids["a"]
                    }:
                        raise RuntimeError(f"{label} RAG context was not authorization-safe.")
                    if {row["document_id"] for row in data["citations"]} != {
                        document_ids["a"]
                    }:
                        raise RuntimeError(f"{label} citations were not authorization-safe.")

                if any(UNAUTHORIZED_SENTINEL in prompt for prompt in captured_prompts):
                    raise RuntimeError("Unauthorized sentinel reached an LLM prompt.")

                revoke_document_read_access(
                    document_ids["a"],
                    user_ids["bob"],
                    user_ids["alice"],
                    session_factory,
                )
                prompts_before_revoke_check = len(captured_prompts)
                bob_after_revoke = client.post(
                    "/chat/",
                    headers=bob_headers,
                    json={
                        "question": "Alice shared policy access fact",
                        "retrieval_mode": "keyword",
                        "top_k": 2,
                        "min_score": 0.0,
                    },
                )
                revoked_data = bob_after_revoke.json()["data"]
                if (
                    bob_after_revoke.status_code != 200
                    or revoked_data["contexts"]
                    or revoked_data["citations"]
                    or revoked_data["answer_mode"] != "no_context"
                    or len(captured_prompts) != prompts_before_revoke_check
                ):
                    raise RuntimeError("ACL revocation did not affect the next request.")

                with patch(
                    "app.services.access_control.get_session_factory",
                    side_effect=SQLAlchemyError("synthetic database outage"),
                ), patch("app.services.vector_store.embed_text") as embed:
                    failed_closed = client.post(
                        "/search",
                        headers=_bearer(user_ids["alice"]),
                        json={"query": "policy"},
                    )
                if failed_closed.status_code != 503 or embed.called:
                    raise RuntimeError("Authorization database failure did not fail closed.")
        finally:
            with session_factory() as session, session.begin():
                session.execute(
                    delete(DocumentACLRecord).where(
                        DocumentACLRecord.document_id.in_(set(document_ids.values()))
                    )
                )
                session.execute(
                    delete(DocumentRecord).where(
                        DocumentRecord.document_id.in_(set(document_ids.values()))
                    )
                )
                session.execute(
                    delete(UserRecord).where(
                        UserRecord.user_id.in_(set(user_ids.values()))
                    )
                )

    if _snapshot(session_factory) != before:
        raise RuntimeError("Permission verifier did not restore the preexisting test state.")

    print(
        "Real PostgreSQL permission-aware retrieval verification passed: "
        "database=enterprise_rag_test, "
        f"revision={revision}, ownership_union=true, acl_union=true, "
        "authenticated_search=true, authenticated_chat=true, "
        "faiss_adaptive_filter=true, pre_llm_filter=true, "
        "context_and_citations_authorized=true, revoke_next_request=true, "
        "database_failure_503=true, exact_cleanup=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
