import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval.bm25 import BM25Index, BM25Retriever
from app.services.knowledge_base import search_bm25_chunks


CHUNKS = [
    {
        "chunk_id": "leave-1",
        "document_id": "leave-doc",
        "filename": "leave-policy.pdf",
        "position": 1,
        "chunk_index": 0,
        "page_number": 7,
        "content": "Policy HR-2026 grants employees twenty days of annual leave.",
        "token_count": 9,
        "metadata": {"department": "people"},
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "chunk_id": "holiday-1",
        "document_id": "holiday-doc",
        "filename": "holiday-policy.txt",
        "position": 1,
        "chunk_index": 0,
        "page_number": None,
        "content": "Holiday requests require manager approval before booking travel.",
        "token_count": 8,
        "metadata": {"department": "people"},
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "chunk_id": "security-1",
        "document_id": "security-doc",
        "filename": "security.docx",
        "position": 1,
        "chunk_index": 0,
        "page_number": None,
        "content": "Security standard SEC-99 requires hardware keys for administrators.",
        "token_count": 8,
        "metadata": {"department": "security"},
        "created_at": "2026-08-01T00:00:00+00:00",
    },
]


class BM25IndexTests(unittest.TestCase):
    def test_index_builds_term_statistics_from_chunks(self):
        index = BM25Index(CHUNKS)

        self.assertEqual(len(index), 3)
        self.assertEqual(index.document_frequency("policy"), 1)
        self.assertEqual(index.document_frequency("security"), 1)
        self.assertGreater(index.average_document_length, 0)

    def test_query_returns_ranked_keyword_matches(self):
        results = BM25Retriever(BM25Index(CHUNKS)).retrieve("SEC-99 hardware keys", top_k=3)

        self.assertEqual(results[0]["chunk_id"], "security-1")
        self.assertGreater(results[0]["bm25_score"], 0)
        self.assertEqual(results[0]["retrieval_mode"], "bm25")

    def test_top_k_limits_results(self):
        results = BM25Retriever(BM25Index(CHUNKS)).retrieve("policy security", top_k=1)

        self.assertEqual(len(results), 1)

    def test_result_preserves_all_chunk_metadata(self):
        result = BM25Retriever(BM25Index(CHUNKS)).retrieve("HR-2026 annual leave", top_k=1)[0]

        self.assertEqual(result["content"], CHUNKS[0]["content"])
        self.assertEqual(result["document_id"], "leave-doc")
        self.assertEqual(result["filename"], "leave-policy.pdf")
        self.assertEqual(result["page_number"], 7)
        self.assertEqual(result["chunk_index"], 0)
        self.assertEqual(result["metadata"], {"department": "people"})

    def test_empty_index_returns_no_results(self):
        results = BM25Retriever(BM25Index([])).retrieve("annual leave", top_k=5)

        self.assertEqual(results, [])

    def test_query_without_indexed_terms_returns_no_results(self):
        results = BM25Retriever(BM25Index(CHUNKS)).retrieve("quantum entanglement", top_k=5)

        self.assertEqual(results, [])

    def test_non_positive_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            BM25Retriever(BM25Index(CHUNKS)).retrieve("policy", top_k=0)

    def test_service_builds_bm25_from_persisted_chunks(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chunks_file = Path(tempdir) / "chunks.json"
            chunks_file.write_text(json.dumps(CHUNKS), encoding="utf-8")

            results = search_bm25_chunks(
                "annual leave",
                top_k=1,
                index_path=chunks_file,
                allowed_document_ids={"leave-doc", "security-doc"},
            )

        self.assertEqual(results[0]["chunk_id"], "leave-1")

    def test_bm25_filters_corpus_before_scoring_and_drops_missing_identity(self):
        blocked = {
            **CHUNKS[0],
            "chunk_id": "blocked-1",
            "document_id": "blocked-doc",
            "content": "SEC-99 hardware keys SEC-99 hardware keys exact exact exact.",
        }
        allowed = {**CHUNKS[2], "content": "SEC-99 hardware keys."}
        missing = {
            key: value
            for key, value in blocked.items()
            if key != "document_id"
        }
        with tempfile.TemporaryDirectory() as tempdir:
            chunks_file = Path(tempdir) / "chunks.json"
            chunks_file.write_text(
                json.dumps([blocked, missing, allowed]),
                encoding="utf-8",
            )
            results = search_bm25_chunks(
                "SEC-99 hardware keys",
                top_k=3,
                index_path=chunks_file,
                allowed_document_ids={"security-doc"},
            )

        self.assertEqual([result["chunk_id"] for result in results], ["security-1"])


if __name__ == "__main__":
    unittest.main()
