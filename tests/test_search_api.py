import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app
from app.auth.dependencies import get_current_user
from app.services.embeddings import EmbeddingClientError
from app.services.access_control import AccessControlUnavailableError
from app.services.user_registry import UserIdentity


TEST_USER = UserIdentity(
    user_id=UUID("00000000-0000-0000-0000-000000000401"),
    username="search-user",
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
ALLOWED_DOCUMENT_IDS = frozenset({"leave-doc"})


class SemanticSearchApiTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_current_user] = lambda: TEST_USER
        self.authorization_patcher = patch(
            "app.routers.search.get_readable_document_ids",
            return_value=ALLOWED_DOCUMENT_IDS,
        )
        self.authorization_patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.authorization_patcher.stop()
        app.dependency_overrides.pop(get_current_user, None)

    def test_search_returns_ranked_chunks_without_llm(self):
        result = {
            "chunk_id": "leave-1",
            "document_id": "leave-doc",
            "filename": "leave-policy.pdf",
            "position": 1,
            "chunk_index": 0,
            "page_number": 7,
            "content": "Employees receive twenty days of annual leave.",
            "token_count": 8,
            "score": 0.91,
            "retrieval_mode": "vector",
            "embedding_model": "test-model",
            "query_embedding_model": "test-model",
            "vector_store_backend": "faiss",
            "created_at": "2026-08-01T00:00:00+00:00",
        }

        with patch("app.routers.search.search_vector_chunks", return_value=[result]) as search:
            response = self.client.post(
                "/search",
                json={"query": "  annual leave days  ", "top_k": 1},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["query"], "annual leave days")
        self.assertEqual(payload["data"]["result_count"], 1)
        self.assertEqual(payload["data"]["results"][0]["page_number"], 7)
        self.assertEqual(payload["data"]["results"][0]["vector_store_backend"], "faiss")
        search.assert_called_once_with(
            "annual leave days",
            top_k=1,
            min_score=0.2,
            allowed_document_ids=ALLOWED_DOCUMENT_IDS,
        )

    def test_search_empty_database_is_successful_empty_result(self):
        with patch("app.routers.search.search_vector_chunks", return_value=[]):
            response = self.client.post(
                "/search",
                json={"query": "annual leave", "top_k": 3},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["result_count"], 0)
        self.assertEqual(response.json()["data"]["results"], [])

    def test_search_rejects_blank_query_and_invalid_top_k(self):
        blank = self.client.post("/search", json={"query": "   "})
        invalid_top_k = self.client.post(
            "/search",
            json={"query": "policy", "top_k": 0},
        )

        self.assertEqual(blank.status_code, 422)
        self.assertEqual(invalid_top_k.status_code, 422)

    def test_search_embedding_failure_returns_safe_503(self):
        with patch(
            "app.routers.search.search_vector_chunks",
            side_effect=EmbeddingClientError("private provider detail"),
        ):
            response = self.client.post("/search", json={"query": "policy"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Semantic search is temporarily unavailable.",
        )
        self.assertNotIn("private provider detail", response.text)

    def test_authorization_database_failure_is_safe_503_before_search(self):
        with patch(
            "app.routers.search.get_readable_document_ids",
            side_effect=AccessControlUnavailableError("private database detail"),
        ), patch("app.routers.search.search_vector_chunks") as search:
            response = self.client.post("/search", json={"query": "policy"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Document authorization is temporarily unavailable.",
        )
        self.assertNotIn("private database detail", response.text)
        search.assert_not_called()

    def test_missing_and_invalid_tokens_return_401_before_search(self):
        app.dependency_overrides.pop(get_current_user, None)
        with patch("app.routers.search.search_vector_chunks") as search:
            missing = self.client.post("/search", json={"query": "policy"})
            invalid = self.client.post(
                "/search",
                headers={"Authorization": "Bearer not-a-valid-token"},
                json={"query": "policy"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
