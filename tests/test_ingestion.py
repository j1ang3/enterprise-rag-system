import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.knowledge_base import index_document
from app.services.text_loader import ExtractedSection


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)
        self.chunks_file = self.temp_path / "chunks.json"
        self.vectors_file = self.temp_path / "vectors.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_page_sections_create_traceable_structured_chunks(self):
        sections = (
            ExtractedSection(text="Page one policy", page_number=1),
            ExtractedSection(text="Page two policy", page_number=2),
        )

        with patch("app.services.knowledge_base.index_vector_chunks") as vector_index:
            chunks = index_document(
                "document-123",
                "policy.pdf",
                "[Page 1]\nPage one policy\n\n[Page 2]\nPage two policy",
                chunk_size=100,
                chunk_overlap=10,
                index_path=self.chunks_file,
                vector_index_path=self.vectors_file,
                sections=sections,
            )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [(chunk["chunk_index"], chunk["position"], chunk["page_number"]) for chunk in chunks],
            [(0, 1, 1), (1, 2, 2)],
        )
        self.assertEqual(chunks[0]["document_id"], "document-123")
        self.assertEqual(chunks[0]["chunk_id"], "document-123-1")
        self.assertEqual(chunks[0]["filename"], "policy.pdf")
        self.assertEqual(chunks[0]["content"], "Page one policy")
        self.assertIn("created_at", chunks[0])
        self.assertEqual(json.loads(self.chunks_file.read_text(encoding="utf-8")), chunks)
        vector_index.assert_called_once_with("document-123", chunks, self.vectors_file)

    def test_plain_text_uses_compatible_position_and_no_page_number(self):
        with patch("app.services.knowledge_base.index_vector_chunks"):
            chunks = index_document(
                "document-plain",
                "policy.txt",
                "abcdefghij",
                chunk_size=6,
                chunk_overlap=2,
                index_path=self.chunks_file,
                vector_index_path=self.vectors_file,
            )

        self.assertEqual([chunk["chunk_index"] for chunk in chunks], [0, 1])
        self.assertEqual([chunk["position"] for chunk in chunks], [1, 2])
        self.assertTrue(all(chunk["page_number"] is None for chunk in chunks))

    def test_reindex_replaces_only_the_same_document(self):
        with patch("app.services.knowledge_base.index_vector_chunks"):
            index_document(
                "doc-a",
                "a.txt",
                "old content",
                index_path=self.chunks_file,
                vector_index_path=self.vectors_file,
            )
            index_document(
                "doc-b",
                "b.txt",
                "other content",
                index_path=self.chunks_file,
                vector_index_path=self.vectors_file,
            )
            index_document(
                "doc-a",
                "a.txt",
                "new content",
                index_path=self.chunks_file,
                vector_index_path=self.vectors_file,
            )

        stored = json.loads(self.chunks_file.read_text(encoding="utf-8"))
        self.assertEqual(len(stored), 2)
        self.assertEqual(
            {chunk["document_id"]: chunk["content"] for chunk in stored},
            {"doc-a": "new content", "doc-b": "other content"},
        )


if __name__ == "__main__":
    unittest.main()
