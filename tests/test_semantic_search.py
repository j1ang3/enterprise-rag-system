import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.services import vector_store


CHUNKS = [
    {
        "chunk_id": "leave-1",
        "document_id": "leave-doc",
        "filename": "leave-policy.pdf",
        "position": 1,
        "chunk_index": 0,
        "page_number": 7,
        "content": "Employees receive twenty days of annual leave.",
        "token_count": 8,
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "chunk_id": "holiday-1",
        "document_id": "holiday-doc",
        "filename": "holiday-policy.txt",
        "position": 1,
        "chunk_index": 0,
        "page_number": None,
        "content": "Holiday requests require manager approval.",
        "token_count": 6,
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "chunk_id": "security-1",
        "document_id": "security-doc",
        "filename": "security.docx",
        "position": 1,
        "chunk_index": 0,
        "page_number": None,
        "content": "Passwords must not be shared.",
        "token_count": 5,
        "created_at": "2026-08-01T00:00:00+00:00",
    },
]


def controlled_embedding(text):
    if "annual leave" in text.lower() or "leave days" in text.lower():
        return [1.0, 0.0, 0.0]
    if "holiday" in text.lower():
        return [0.8, 0.2, 0.0]
    return [0.0, 1.0, 0.0]


class SemanticSearchTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = settings.vector_store_backend
        settings.vector_store_backend = "faiss"
        self.tempdir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.tempdir.name) / "vectors.json"

    def tearDown(self):
        settings.vector_store_backend = self.previous_backend
        self.tempdir.cleanup()

    def rebuild(self):
        with patch.object(vector_store, "embed_text", side_effect=controlled_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="controlled-model",
        ):
            return vector_store.rebuild_vector_index(CHUNKS, self.index_path)

    def search(self, query, top_k=5, min_score=0.0):
        with patch.object(vector_store, "embed_text", side_effect=controlled_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="controlled-model",
        ):
            return vector_store.search_vector_chunks(
                query,
                top_k=top_k,
                index_path=self.index_path,
                min_score=min_score,
                allowed_document_ids={chunk["document_id"] for chunk in CHUNKS},
            )

    def test_faiss_storage_persists_embedding_text_and_metadata(self):
        self.rebuild()

        stored = json.loads(self.index_path.read_text(encoding="utf-8"))
        leave = stored[0]

        self.assertTrue(vector_store._faiss_index_path(self.index_path).exists())
        self.assertTrue(vector_store._faiss_metadata_path(self.index_path).exists())
        self.assertEqual(leave["embedding"], [1.0, 0.0, 0.0])
        self.assertEqual(leave["content"], CHUNKS[0]["content"])
        self.assertEqual(leave["page_number"], 7)
        self.assertEqual(leave["chunk_index"], 0)
        self.assertEqual(leave["token_count"], 8)

    def test_search_honors_top_k_and_ranks_controlled_relevance(self):
        self.rebuild()

        results = self.search("How many leave days are available?", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "leave-1")
        self.assertEqual(results[1]["chunk_id"], "holiday-1")
        self.assertEqual(results[0]["vector_store_backend"], "faiss")

    def test_empty_vector_database_returns_empty_results(self):
        results = self.search("annual leave", top_k=3)

        self.assertEqual(results, [])

    def test_retrieved_result_preserves_source_metadata(self):
        self.rebuild()

        result = self.search("annual leave", top_k=1)[0]

        self.assertEqual(result["document_id"], "leave-doc")
        self.assertEqual(result["filename"], "leave-policy.pdf")
        self.assertEqual(result["position"], 1)
        self.assertEqual(result["chunk_index"], 0)
        self.assertEqual(result["page_number"], 7)
        self.assertEqual(result["token_count"], 8)
        self.assertEqual(result["embedding_model"], "controlled-model")
        self.assertEqual(result["query_embedding_model"], "controlled-model")


if __name__ == "__main__":
    unittest.main()
