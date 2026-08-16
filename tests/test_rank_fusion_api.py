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
    user_id=UUID("00000000-0000-0000-0000-000000000403"),
    username="fusion-user",
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)
ALLOWED_DOCUMENT_IDS = frozenset({"policy-doc"})


class RankFusionSearchApiTests(unittest.TestCase):
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

    def test_fused_hybrid_search_returns_rrf_results(self):
        fused_result = {
            "chunk_id": "shared",
            "document_id": "policy-doc",
            "filename": "policy.pdf",
            "position": 1,
            "chunk_index": 0,
            "page_number": 3,
            "content": "Policy HR-2026",
            "text": "Policy HR-2026",
            "metadata": {
                "document_id": "policy-doc",
                "filename": "policy.pdf",
                "page_number": 3,
            },
            "source": "rrf",
            "score": 0.0325,
            "fused_score": 0.0325,
            "retrieval_mode": "rrf",
            "matched_sources": ["vector", "bm25"],
            "source_ranks": {"vector": 2, "bm25": 1},
            "source_scores": {"vector": 0.83, "bm25": 7.41},
        }

        with patch(
            "app.routers.search.search_fused_hybrid_chunks",
            return_value=[fused_result],
        ) as search:
            response = self.client.post(
                "/search/hybrid/fused",
                json={
                    "query": "  HR-2026 policy  ",
                    "top_k": 2,
                    "candidate_depth": 5,
                    "rrf_k": 60,
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["data"]["query"], "HR-2026 policy")
        self.assertEqual(payload["data"]["top_k"], 2)
        self.assertEqual(payload["data"]["candidate_depth"], 5)
        self.assertEqual(payload["data"]["rrf_k"], 60)
        self.assertEqual(payload["data"]["results"][0]["chunk_id"], "shared")
        self.assertEqual(
            payload["data"]["results"][0]["matched_sources"],
            ["vector", "bm25"],
        )
        search.assert_called_once_with(
            "HR-2026 policy",
            top_k=2,
            candidate_depth=5,
            rrf_k=60,
            allowed_document_ids=ALLOWED_DOCUMENT_IDS,
        )

    def test_fused_hybrid_search_validates_fusion_parameters(self):
        invalid_candidate_depth = self.client.post(
            "/search/hybrid/fused",
            json={"query": "policy", "candidate_depth": 0},
        )
        invalid_rrf_k = self.client.post(
            "/search/hybrid/fused",
            json={"query": "policy", "rrf_k": 0},
        )

        self.assertEqual(invalid_candidate_depth.status_code, 422)
        self.assertEqual(invalid_rrf_k.status_code, 422)

    def test_fused_hybrid_search_embedding_failure_returns_safe_503(self):
        with patch(
            "app.routers.search.search_fused_hybrid_chunks",
            side_effect=EmbeddingClientError("private provider detail"),
        ):
            response = self.client.post(
                "/search/hybrid/fused",
                json={"query": "policy"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Rank-fused hybrid search is temporarily unavailable.",
        )
        self.assertNotIn("private provider detail", response.text)


if __name__ == "__main__":
    unittest.main()
