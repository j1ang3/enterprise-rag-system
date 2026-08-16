import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app
from app.auth.dependencies import get_current_user
from app.services.storage_paths import DocumentStoragePaths
from app.services.user_registry import UserIdentity
from tests.document_fixtures import build_docx_bytes, write_text_pdf


class DocumentPipelineTests(unittest.TestCase):
    def setUp(self):
        authenticated_user = UserIdentity(
            user_id=UUID("00000000-0000-0000-0000-000000000202"),
            username="pipeline-test-user",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        app.dependency_overrides[get_current_user] = lambda: authenticated_user
        self.addCleanup(app.dependency_overrides.pop, get_current_user, None)
        self.client = TestClient(app)
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)
        self.storage_paths = DocumentStoragePaths(
            upload_dir=self.temp_path / "uploads",
            text_dir=self.temp_path / "texts",
            index_dir=self.temp_path / "index",
            chunks_file=self.temp_path / "index" / "chunks.json",
            vectors_file=self.temp_path / "index" / "vectors.json",
        )
        self.read_authorization_patcher = patch(
            "app.routers.documents.can_user_read_document",
            return_value=True,
        )
        self.read_authorization_patcher.start()
        self.addCleanup(self.read_authorization_patcher.stop)

    def tearDown(self):
        self.tempdir.cleanup()

    def _pipeline_patches(self):
        return (
            patch(
                "app.routers.documents.get_document_storage_paths",
                return_value=self.storage_paths,
            ),
            patch(
                "app.services.knowledge_base.get_document_storage_paths",
                return_value=self.storage_paths,
            ),
            patch("app.services.knowledge_base.index_vector_chunks"),
            patch("app.routers.documents.verify_document_registry_available"),
            patch("app.routers.documents.register_document"),
        )

    def test_pdf_upload_runs_parser_chunker_and_page_metadata_pipeline(self):
        pdf_path = self.temp_path / "source.pdf"
        write_text_pdf(pdf_path, ["Page one policy", "Page two policy"])
        router_paths, knowledge_paths, vector_index, registry_check, registry_write = (
            self._pipeline_patches()
        )

        with router_paths, knowledge_paths, vector_index, registry_check, registry_write:
            upload = self.client.post(
                "/documents/upload",
                files={"file": ("policy.pdf", pdf_path.read_bytes(), "application/pdf")},
            )
            document_id = upload.json()["data"]["document_id"]
            chunks = self.client.get(f"/documents/{document_id}/chunks")

        self.assertEqual(upload.status_code, 200)
        self.assertIn("[Page 1]", upload.json()["data"]["preview"])
        self.assertEqual(chunks.status_code, 200)
        chunk_items = chunks.json()["data"]["chunks"]
        self.assertEqual([chunk["page_number"] for chunk in chunk_items], [1, 2])
        self.assertEqual([chunk["chunk_index"] for chunk in chunk_items], [0, 1])
        self.assertEqual(
            [chunk["content"] for chunk in chunk_items],
            ["Page one policy", "Page two policy"],
        )

    def test_docx_upload_runs_parser_and_chunker_pipeline(self):
        router_paths, knowledge_paths, vector_index, registry_check, registry_write = (
            self._pipeline_patches()
        )

        with router_paths, knowledge_paths, vector_index, registry_check, registry_write:
            upload = self.client.post(
                "/documents/upload",
                files={
                    "file": (
                        "policy.docx",
                        build_docx_bytes("DOCX annual leave policy"),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
            document_id = upload.json()["data"]["document_id"]
            chunks = self.client.get(f"/documents/{document_id}/chunks")

        self.assertEqual(upload.status_code, 200)
        self.assertIn("DOCX annual leave policy", upload.json()["data"]["preview"])
        self.assertEqual(chunks.status_code, 200)
        chunk_item = chunks.json()["data"]["chunks"][0]
        self.assertEqual(chunk_item["content"], "DOCX annual leave policy")
        self.assertIsNone(chunk_item["page_number"])


if __name__ == "__main__":
    unittest.main()
