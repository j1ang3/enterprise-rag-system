import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app
from app.auth.dependencies import get_current_user
from app.services.embeddings import EmbeddingClientError
from app.services.user_registry import UserIdentity


TEST_USER = UserIdentity(
    user_id=UUID("00000000-0000-0000-0000-000000000402"),
    username="hybrid-user",
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
ALLOWED_DOCUMENT_IDS = frozenset({"leave-doc"})


class HybridSearchApiTests(unittest.TestCase):
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

    def test_hybrid_search_returns_combined_candidates_without_llm(self):
        candidates = [
            {
                "chunk_id": "leave-vector",
                "document_id": "leave-doc",
                "filename": "leave.pdf",
                "position": 1,
                "chunk_index": 0,
                "page_number": 7,
                "content": "Annual leave policy",
                "text": "Annual leave policy",
                "metadata": {
                    "document_id": "leave-doc",
                    "filename": "leave.pdf",
                    "page_number": 7,
                },
                "source": "vector",
                "score": 0.91,
                "retrieval_mode": "vector",
            },
            {
                "chunk_id": "leave-bm25",
                "document_id": "leave-doc",
                "filename": "leave.pdf",
                "position": 2,
                "chunk_index": 1,
                "page_number": 8,
                "content": "Policy code HR-2026",
                "text": "Policy code HR-2026",
                "metadata": {
                    "document_id": "leave-doc",
                    "filename": "leave.pdf",
                    "page_number": 8,
                },
                "source": "bm25",
                "score": 2.4,
                "bm25_score": 2.4,
                "retrieval_mode": "bm25",
            },
        ]

        with patch(
            "app.routers.search.search_hybrid_chunks",
            return_value=candidates,
        ) as search:
            response = self.client.post(
                "/search/hybrid",
                json={"query": "  HR-2026 leave  ", "top_k": 2},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["data"]["query"], "HR-2026 leave")
        self.assertEqual(payload["data"]["top_k_per_source"], 2)
        self.assertEqual(payload["data"]["result_count"], 2)
        self.assertEqual(
            [item["source"] for item in payload["data"]["results"]],
            ["vector", "bm25"],
        )
        search.assert_called_once_with(
            "HR-2026 leave",
            top_k=2,
            allowed_document_ids=ALLOWED_DOCUMENT_IDS,
        )

    def test_hybrid_search_rejects_blank_query_and_invalid_top_k(self):
        blank = self.client.post("/search/hybrid", json={"query": "   "})
        invalid_top_k = self.client.post(
            "/search/hybrid",
            json={"query": "policy", "top_k": 0},
        )

        self.assertEqual(blank.status_code, 422)
        self.assertEqual(invalid_top_k.status_code, 422)

    def test_hybrid_search_embedding_failure_returns_safe_503(self):
        with patch(
            "app.routers.search.search_hybrid_chunks",
            side_effect=EmbeddingClientError("private provider detail"),
        ):
            response = self.client.post(
                "/search/hybrid",
                json={"query": "policy"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Hybrid search is temporarily unavailable.",
        )
        self.assertNotIn("private provider detail", response.text)


if __name__ == "__main__":
    unittest.main()
