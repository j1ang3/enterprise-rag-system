import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import jwt
import numpy as np
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, delete, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.tokens import JWT_ALGORITHM, create_access_token
from app.core.config import settings
from app.db.base import Base
from app.db.models import DocumentACLRecord, UserRecord
from app.main import app
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever
from app.services import vector_store
from app.services.access_control import (
    AccessControlUnavailableError,
    can_user_read_document,
    get_readable_document_ids,
    grant_document_read_access,
    revoke_document_read_access,
)
from app.services.document_registry import DocumentRegistration, register_document
from app.services.knowledge_base import search_bm25_chunks
from app.services.rag_service import answer_question
from app.services.search_service import RerankedHybridConfig
from app.services.storage_paths import DocumentStoragePaths
from app.services.user_registry import create_user


ENCODED_TEST_HASH = "$argon2id$w10-t7-test-only-encoded-value"
TEST_SECRET = "w10-t7-test-only-jwt-secret-with-more-than-64-bytes-of-material"
TEST_MODEL = "w10-t7-controlled-embedding"
SENTINEL = "W10T7_UNAUTHORIZED_ONLY_7F42C9_SENTINEL"
ALICE_ID = UUID("00000000-0000-0000-0000-000000000701")


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
        file_size_bytes=32,
        upload_path=None,
        text_path=None,
        chunk_count=1,
        owner_id=owner_id,
        created_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )


def _chunk(
    chunk_id: str,
    document_id: str | None,
    content: str,
    *,
    position: int = 1,
) -> dict:
    chunk = {
        "chunk_id": chunk_id,
        "filename": f"{document_id or 'missing'}.txt",
        "position": position,
        "chunk_index": position - 1,
        "page_number": None,
        "content": content,
        "token_count": len(content.split()),
        "created_at": "2026-08-09T00:00:00+00:00",
    }
    if document_id is not None:
        chunk["document_id"] = document_id
    return chunk


def _vector_entry(chunk: dict, score: float) -> dict:
    return {
        **chunk,
        "embedding": [score, 0.0],
        "embedding_model": TEST_MODEL,
    }


class PermissionDecisionMatrixTests(unittest.TestCase):
    def setUp(self):
        self.engine = _sqlite_engine()
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self.users = {
            name: create_user(
                f"w10t7-{name}",
                self.session_factory,
                password_hash=ENCODED_TEST_HASH,
            )
            for name in ("alice", "bob", "carol", "dave")
        }
        for document_id, owner_name in (("doc-a", "alice"), ("doc-b", "bob"), ("doc-c", "carol")):
            register_document(
                _registration(document_id, self.users[owner_name].user_id),
                self.session_factory,
            )
        grant_document_read_access(
            "doc-a",
            self.users["bob"].user_id,
            self.users["alice"].user_id,
            self.session_factory,
        )

    def tearDown(self):
        self.engine.dispose()

    def test_alice_bob_carol_owner_shared_and_denied_nine_cell_matrix(self):
        expected = {
            ("alice", "doc-a"): True,
            ("alice", "doc-b"): False,
            ("alice", "doc-c"): False,
            ("bob", "doc-a"): True,
            ("bob", "doc-b"): True,
            ("bob", "doc-c"): False,
            ("carol", "doc-a"): False,
            ("carol", "doc-b"): False,
            ("carol", "doc-c"): True,
        }

        actual = {
            (principal, document_id): can_user_read_document(
                self.users[principal].user_id,
                document_id,
                self.session_factory,
            )
            for principal, document_id in expected
        }

        self.assertEqual(actual, expected)

    def test_bulk_readable_sets_match_matrix_and_owner_acl_overlap_deduplicates(self):
        with self.session_factory() as session, session.begin():
            session.add(
                DocumentACLRecord(
                    document_id="doc-a",
                    user_id=self.users["alice"].user_id,
                    created_at=datetime.now(timezone.utc),
                )
            )

        self.assertEqual(
            get_readable_document_ids(self.users["alice"].user_id, self.session_factory),
            frozenset({"doc-a"}),
        )
        self.assertEqual(
            get_readable_document_ids(self.users["bob"].user_id, self.session_factory),
            frozenset({"doc-a", "doc-b"}),
        )
        self.assertEqual(
            get_readable_document_ids(self.users["carol"].user_id, self.session_factory),
            frozenset({"doc-c"}),
        )

    def test_dave_with_no_owned_or_shared_documents_is_explicitly_deny_all(self):
        self.assertEqual(
            get_readable_document_ids(self.users["dave"].user_id, self.session_factory),
            frozenset(),
        )
        for document_id in ("doc-a", "doc-b", "doc-c"):
            with self.subTest(document_id=document_id):
                self.assertFalse(
                    can_user_read_document(
                        self.users["dave"].user_id,
                        document_id,
                        self.session_factory,
                    )
                )


class DocumentAccessAuthorizationApiTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides.clear()
        self.engine = _sqlite_engine()
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self.users = {
            name: create_user(
                f"w11t6-document-{name}",
                self.session_factory,
                password_hash=ENCODED_TEST_HASH,
            )
            for name in ("alice", "bob", "carol", "dave")
        }
        for document_id, owner_name in (
            ("doc-a", "alice"),
            ("doc-b", "bob"),
            ("doc-c", "carol"),
        ):
            register_document(
                _registration(document_id, self.users[owner_name].user_id),
                self.session_factory,
            )
        grant_document_read_access(
            "doc-a",
            self.users["bob"].user_id,
            self.users["alice"].user_id,
            self.session_factory,
        )

        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        upload_dir = root / "uploads"
        text_dir = root / "texts"
        index_dir = root / "index"
        upload_dir.mkdir()
        text_dir.mkdir()
        index_dir.mkdir()
        self.storage_paths = DocumentStoragePaths(
            upload_dir=upload_dir,
            text_dir=text_dir,
            index_dir=index_dir,
            chunks_file=index_dir / "chunks.json",
            vectors_file=index_dir / "vectors.json",
        )
        self.contents = {
            "doc-a": "ALICE_PRIVATE_DOCUMENT_CONTENT",
            "doc-b": "BOB_PRIVATE_DOCUMENT_CONTENT",
            "doc-c": SENTINEL,
        }
        for document_id, content in self.contents.items():
            (text_dir / f"{document_id}.txt").write_text(content, encoding="utf-8")
        self.chunks = {
            document_id: [
                _chunk(
                    f"{document_id}-chunk",
                    document_id,
                    content,
                )
            ]
            for document_id, content in self.contents.items()
        }

        self.patchers = (
            patch(
                "app.services.user_registry.get_session_factory",
                return_value=self.session_factory,
            ),
            patch(
                "app.services.access_control.get_session_factory",
                return_value=self.session_factory,
            ),
            patch(
                "app.services.document_registry.get_session_factory",
                return_value=self.session_factory,
            ),
            patch.object(settings, "jwt_secret_key", SecretStr(TEST_SECRET)),
            patch(
                "app.routers.documents.get_document_storage_paths",
                return_value=self.storage_paths,
            ),
            patch(
                "app.routers.documents.get_document_chunks",
                side_effect=lambda document_id: self.chunks.get(document_id, []),
            ),
        )
        started = [patcher.start() for patcher in self.patchers]
        self.storage_paths_mock = started[-2]
        self.chunk_lookup = started[-1]
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.tempdir.cleanup()
        self.engine.dispose()

    def _headers(self, username: str) -> dict[str, str]:
        token = create_access_token(self.users[username].user_id).value
        return {"Authorization": f"Bearer {token}"}

    def _list_document_ids(
        self,
        username: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> set[str]:
        response = self.client.get(
            "/documents/",
            headers=headers or self._headers(username),
        )
        self.assertEqual(response.status_code, 200)
        return {
            document["document_id"]
            for document in response.json()["data"]["documents"]
        }

    def test_anonymous_document_read_endpoints_are_denied(self):
        responses = (
            self.client.get("/documents/"),
            self.client.get("/documents/doc-a/preview"),
            self.client.get("/documents/doc-a/chunks"),
        )

        self.assertEqual([response.status_code for response in responses], [401] * 3)
        self.assertTrue(
            all(
                self.contents["doc-a"] not in response.text
                for response in responses
            )
        )

    def test_list_returns_only_owned_and_explicitly_shared_documents(self):
        expected = {
            "alice": {"doc-a"},
            "bob": {"doc-a", "doc-b"},
            "carol": {"doc-c"},
            "dave": set(),
        }

        actual = {
            username: self._list_document_ids(username)
            for username in expected
        }

        self.assertEqual(actual, expected)

    def test_owner_and_shared_reader_can_read_preview_and_chunks(self):
        for username in ("alice", "bob"):
            with self.subTest(username=username):
                headers = self._headers(username)
                preview = self.client.get("/documents/doc-a/preview", headers=headers)
                chunks = self.client.get("/documents/doc-a/chunks", headers=headers)

                self.assertEqual(preview.status_code, 200)
                self.assertEqual(chunks.status_code, 200)
                self.assertIn(self.contents["doc-a"], preview.text)
                self.assertIn(self.contents["doc-a"], chunks.text)

    def test_unrelated_authenticated_user_is_denied_before_content_access(self):
        headers = self._headers("carol")
        self.storage_paths_mock.reset_mock()
        self.chunk_lookup.reset_mock()

        preview = self.client.get("/documents/doc-a/preview", headers=headers)
        chunks = self.client.get("/documents/doc-a/chunks", headers=headers)

        self.assertEqual(preview.status_code, 403)
        self.assertEqual(chunks.status_code, 403)
        self.assertNotIn(self.contents["doc-a"], preview.text)
        self.assertNotIn(self.contents["doc-a"], chunks.text)
        self.storage_paths_mock.assert_not_called()
        self.chunk_lookup.assert_not_called()

    def test_revoked_reader_loses_list_preview_and_chunk_access_immediately(self):
        headers = self._headers("bob")
        self.assertEqual(
            self._list_document_ids("bob", headers=headers),
            {"doc-a", "doc-b"},
        )
        self.assertEqual(
            self.client.get("/documents/doc-a/preview", headers=headers).status_code,
            200,
        )
        self.assertEqual(
            self.client.get("/documents/doc-a/chunks", headers=headers).status_code,
            200,
        )

        revoke_document_read_access(
            "doc-a",
            self.users["bob"].user_id,
            self.users["alice"].user_id,
            self.session_factory,
        )

        self.assertEqual(
            self._list_document_ids("bob", headers=headers),
            {"doc-b"},
        )
        preview = self.client.get("/documents/doc-a/preview", headers=headers)
        chunks = self.client.get("/documents/doc-a/chunks", headers=headers)
        self.assertEqual(preview.status_code, 403)
        self.assertEqual(chunks.status_code, 403)
        self.assertNotIn(self.contents["doc-a"], preview.text)
        self.assertNotIn(self.contents["doc-a"], chunks.text)

    def test_authenticated_missing_document_preserves_not_found_semantics(self):
        headers = self._headers("alice")

        preview = self.client.get("/documents/missing/preview", headers=headers)
        chunks = self.client.get("/documents/missing/chunks", headers=headers)

        self.assertEqual(preview.status_code, 404)
        self.assertEqual(chunks.status_code, 404)


class RetrievalAuthenticationNegativeTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides.clear()
        self.engine = _sqlite_engine()
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self.alice = create_user(
            "w10t7-auth-alice",
            self.session_factory,
            password_hash=ENCODED_TEST_HASH,
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
        app.dependency_overrides.clear()
        self.secret_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()

    def test_missing_invalid_expired_unknown_and_deleted_principals_stop_before_retrieval(self):
        now = datetime.now(timezone.utc)
        expired = jwt.encode(
            {
                "sub": str(self.alice.user_id),
                "iat": now - timedelta(minutes=2),
                "exp": now - timedelta(minutes=1),
            },
            TEST_SECRET,
            algorithm=JWT_ALGORITHM,
        )
        unknown = create_access_token(uuid4()).value
        deleted = create_access_token(self.alice.user_id).value
        with self.session_factory() as session, session.begin():
            session.execute(
                delete(UserRecord).where(UserRecord.user_id == self.alice.user_id)
            )

        headers_by_case = {
            "missing": {},
            "invalid": {"Authorization": "Bearer malformed"},
            "expired": {"Authorization": f"Bearer {expired}"},
            "unknown": {"Authorization": f"Bearer {unknown}"},
            "deleted": {"Authorization": f"Bearer {deleted}"},
        }
        with patch("app.routers.search.get_readable_document_ids") as authorize, patch(
            "app.routers.search.search_vector_chunks"
        ) as retrieve:
            responses = {
                name: self.client.post(
                    "/search",
                    headers=headers,
                    json={"query": "synthetic policy"},
                )
                for name, headers in headers_by_case.items()
            }

        self.assertEqual(
            {name: response.status_code for name, response in responses.items()},
            {name: 401 for name in headers_by_case},
        )
        authorize.assert_not_called()
        retrieve.assert_not_called()


class PermissionAwareBackendMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.chunks_path = root / "chunks.json"
        self.vectors_path = root / "vectors.json"
        self.allowed_chunks = [
            _chunk("a-1", "doc-a", "authorized weak policy term", position=1),
            _chunk("a-2", "doc-a", "authorized lower policy term", position=2),
        ]
        self.blocked_chunks = [
            _chunk(
                "c-1",
                "doc-c",
                f"policy policy policy strongest {SENTINEL}",
                position=1,
            ),
            _chunk(
                "c-2",
                "doc-c",
                f"policy policy second strongest {SENTINEL}",
                position=2,
            ),
        ]
        self.missing_identity = _chunk(
            "missing-1",
            None,
            "policy unknown identity",
        )
        self.chunks_path.write_text(
            json.dumps([*self.allowed_chunks, *self.blocked_chunks, self.missing_identity]),
            encoding="utf-8",
        )
        vector_store._save_vectors(
            [
                _vector_entry(self.blocked_chunks[0], 0.99),
                _vector_entry(self.blocked_chunks[1], 0.98),
                _vector_entry(self.missing_identity, 0.97),
                _vector_entry(self.allowed_chunks[0], 0.60),
                _vector_entry(self.allowed_chunks[1], 0.50),
            ],
            self.vectors_path,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _embedding_patches(self):
        return (
            patch.object(vector_store, "embed_text", return_value=[1.0, 0.0]),
            patch.object(
                vector_store,
                "embedding_model_for_vector",
                return_value=TEST_MODEL,
            ),
        )

    def test_faiss_raw_strong_matches_are_filtered_and_authorized_exhaustion_is_safe(self):
        loaded = vector_store._load_faiss_index(self.vectors_path)
        self.assertIsNotNone(loaded, "FAISS is a required local backend for W10-T7")
        index, metadata = loaded
        _scores, indexes = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 3)
        raw_document_ids = [metadata[int(position)].get("document_id") for position in indexes[0]]
        self.assertEqual(raw_document_ids, ["doc-c", "doc-c", None])

        embed_patch, model_patch = self._embedding_patches()
        with patch.object(settings, "vector_store_backend", "faiss"), embed_patch, model_patch:
            results = vector_store.search_vector_chunks(
                "policy",
                top_k=3,
                index_path=self.vectors_path,
                min_score=0.0,
                allowed_document_ids={"doc-a"},
            )

        self.assertEqual([result["chunk_id"] for result in results], ["a-1", "a-2"])
        self.assertLess(len(results), 3)
        self.assertTrue(all(result["document_id"] == "doc-a" for result in results))

    def test_vectors_json_strong_matches_and_missing_identity_fail_closed(self):
        embed_patch, model_patch = self._embedding_patches()
        with patch.object(settings, "vector_store_backend", "json"), embed_patch, model_patch:
            results = vector_store.search_vector_chunks(
                "policy",
                top_k=3,
                index_path=self.vectors_path,
                min_score=0.0,
                allowed_document_ids={"doc-a"},
            )

        self.assertEqual([result["chunk_id"] for result in results], ["a-1", "a-2"])
        self.assertNotIn(SENTINEL, json.dumps(results))

    def test_bm25_strong_unauthorized_match_is_excluded_before_ranking(self):
        results = search_bm25_chunks(
            "policy",
            top_k=3,
            index_path=self.chunks_path,
            allowed_document_ids={"doc-a"},
        )

        self.assertEqual({result["document_id"] for result in results}, {"doc-a"})
        self.assertNotIn(SENTINEL, json.dumps(results))

    def test_hybrid_reverse_branch_scenarios_exclude_unauthorized_from_rrf_input(self):
        allowed_vector = {**self.allowed_chunks[0], "score": 0.20}
        allowed_bm25 = {**self.allowed_chunks[1], "score": 0.30, "bm25_score": 0.30}
        blocked_vector = {**self.blocked_chunks[0], "score": 0.99}
        blocked_bm25 = {**self.blocked_chunks[1], "score": 9.99, "bm25_score": 9.99}

        class StubRetriever:
            def __init__(self, source, results):
                self.source = source
                self.results = results

            def retrieve(self, _query, top_k=5):
                return self.results[:top_k]

        scenarios = {
            "faiss_unauthorized_strong": (
                [blocked_vector, allowed_vector],
                [allowed_bm25],
            ),
            "bm25_unauthorized_strong": (
                [allowed_vector],
                [blocked_bm25, allowed_bm25],
            ),
        }
        for scenario, (vector_results, bm25_results) in scenarios.items():
            with self.subTest(scenario=scenario), patch(
                "app.retrieval.hybrid.reciprocal_rank_fusion",
                wraps=reciprocal_rank_fusion,
            ) as fusion:
                results = HybridRetriever(
                    (
                        StubRetriever("vector", vector_results),
                        StubRetriever("bm25", bm25_results),
                    ),
                    allowed_document_ids={"doc-a"},
                ).retrieve_fused(
                    "policy",
                    top_k=2,
                    candidate_depth=2,
                    rrf_k=60,
                )

            rrf_sources = fusion.call_args.args[0]
            rrf_candidates = [candidate for _source, candidates in rrf_sources for candidate in candidates]
            self.assertTrue(rrf_candidates)
            self.assertTrue(
                all(candidate["document_id"] == "doc-a" for candidate in rrf_candidates)
            )
            self.assertTrue(all(result["document_id"] == "doc-a" for result in results))
            self.assertNotIn(SENTINEL, json.dumps(rrf_candidates))

    def test_reranker_context_citations_and_llm_prompt_are_authorized_only(self):
        allowed = {**self.allowed_chunks[0], "score": 0.20}
        blocked = {**self.blocked_chunks[0], "score": 0.99}
        missing = {**self.missing_identity, "score": 0.98}

        class FaultyHybrid:
            def retrieve_fused(self, _query, top_k, *, candidate_depth, rrf_k):
                self.configuration = (candidate_depth, rrf_k)
                return [blocked, missing, allowed][:top_k]

        class CapturingReranker:
            def __init__(self):
                self.candidates = []

            def rerank(self, _query, candidates, top_k=None):
                self.candidates = list(candidates)
                return self.candidates[:top_k]

        reranker = CapturingReranker()
        captured_messages = []

        def fake_completion(messages, **_kwargs):
            captured_messages.extend(messages)
            return "Deterministic authorized answer."

        with patch(
            "app.services.rag_service.get_readable_document_ids",
            return_value=frozenset({"doc-a"}),
        ), patch(
            "app.services.search_service.build_hybrid_retriever",
            return_value=FaultyHybrid(),
        ), patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch(
            "app.services.knowledge_base.chat_completion",
            side_effect=fake_completion,
        ):
            result = answer_question(
                "policy",
                1,
                user_id=ALICE_ID,
                retrieval_mode="hybrid_rerank",
                index_path=self.chunks_path,
                vector_index_path=self.vectors_path,
                reranked_hybrid_config=RerankedHybridConfig(
                    per_source_candidate_depth=3,
                    rerank_candidate_count=3,
                    final_top_k=1,
                ),
                reranker=reranker,
            )

        self.assertEqual({item["document_id"] for item in reranker.candidates}, {"doc-a"})
        self.assertEqual({item["document_id"] for item in result["contexts"]}, {"doc-a"})
        self.assertEqual({item["document_id"] for item in result["citations"]}, {"doc-a"})
        self.assertNotIn(SENTINEL, json.dumps(captured_messages, ensure_ascii=False))

    def test_zero_access_is_deny_all_without_embedding_or_llm(self):
        with patch(
            "app.services.rag_service.get_readable_document_ids",
            return_value=frozenset(),
        ), patch.object(vector_store, "embed_text") as embed, patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch("app.services.knowledge_base.chat_completion") as completion:
            result = answer_question(
                "policy",
                3,
                user_id=ALICE_ID,
                retrieval_mode="vector",
                index_path=self.chunks_path,
                vector_index_path=self.vectors_path,
                min_score=0.0,
            )

        self.assertEqual(result["contexts"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["answer_mode"], "no_context")
        embed.assert_not_called()
        completion.assert_not_called()


class PermissionFailClosedTests(unittest.TestCase):
    def test_authorization_database_failure_stops_retrieval_and_generation(self):
        with patch(
            "app.services.rag_service.get_readable_document_ids",
            side_effect=AccessControlUnavailableError("synthetic outage"),
        ), patch("app.services.rag_service.search_chunks") as retrieve, patch(
            "app.services.rag_service.build_answer"
        ) as generate:
            with self.assertRaises(AccessControlUnavailableError):
                answer_question(
                    "policy",
                    3,
                    user_id=ALICE_ID,
                    retrieval_mode="keyword",
                )

        retrieve.assert_not_called()
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
