import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, patch
from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app
from app.auth.dependencies import get_current_user
from app.security.defenses import SAFE_BLOCKED_RESPONSE
from app.services.document_registry import DocumentRegistryUnavailableError
from app.services.access_control import AccessControlUnavailableError
from app.services.embeddings import EmbeddingClientError
from app.services.llm_client import LLMClientError
from app.services.storage_paths import DocumentStoragePaths
from app.services.user_registry import UserIdentity


AUTHENTICATED_USER = UserIdentity(
    user_id=UUID("00000000-0000-0000-0000-000000000201"),
    username="api-test-user",
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
)


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_current_user] = lambda: AUTHENTICATED_USER
        self.addCleanup(app.dependency_overrides.pop, get_current_user, None)
        self.client = TestClient(app)
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)
        self.upload_dir = self.temp_path / "uploads"
        self.text_dir = self.temp_path / "texts"
        self.upload_dir.mkdir()
        self.text_dir.mkdir()
        self.storage_paths = DocumentStoragePaths(
            upload_dir=self.upload_dir,
            text_dir=self.text_dir,
            index_dir=self.temp_path / "index",
            chunks_file=self.temp_path / "index" / "chunks.json",
            vectors_file=self.temp_path / "index" / "vectors.json",
        )
        self.registry_preflight_patcher = patch(
            "app.routers.documents.verify_document_registry_available"
        )
        self.registry_write_patcher = patch(
            "app.routers.documents.register_document"
        )
        self.registry_preflight = self.registry_preflight_patcher.start()
        self.registry_write = self.registry_write_patcher.start()
        self.authorization_patcher = patch(
            "app.services.rag_service.get_readable_document_ids",
            return_value=frozenset({"doc"}),
        )
        self.document_list_authorization_patcher = patch(
            "app.routers.documents.get_readable_document_ids",
            return_value=frozenset({"doc", "doc-preview", "missing"}),
        )
        self.document_read_authorization_patcher = patch(
            "app.routers.documents.can_user_read_document",
            return_value=True,
        )
        self.authorization_patcher.start()
        self.document_list_authorization = self.document_list_authorization_patcher.start()
        self.document_read_authorization_patcher.start()
        self.addCleanup(self.registry_preflight_patcher.stop)
        self.addCleanup(self.registry_write_patcher.stop)
        self.addCleanup(self.authorization_patcher.stop)
        self.addCleanup(self.document_list_authorization_patcher.stop)
        self.addCleanup(self.document_read_authorization_patcher.stop)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_health_check(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_root_returns_running_message(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "Enterprise RAG System is running."},
        )

    def test_swagger_ui_is_available(self):
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("swagger-ui", response.text.lower())

    def test_chat_requires_question(self):
        response = self.client.post("/chat/", json={"top_k": 2})

        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_invalid_top_k(self):
        response = self.client.post(
            "/chat/",
            json={"question": "What is the policy?", "top_k": 0},
        )

        self.assertEqual(response.status_code, 422)

    def test_chat_rejects_whitespace_only_question(self):
        response = self.client.post("/chat/", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

    def test_chat_normalizes_question_whitespace(self):
        result = {
            "question": "What is the policy?",
            "retrieval_mode": "keyword",
            "min_score": 0.2,
            "answer": "Normalized.",
            "answer_mode": "local_fallback",
            "model": None,
            "llm_error": None,
            "citations": [],
            "contexts": [],
        }

        with patch("app.routers.chat.answer_question", return_value=result) as answer_question:
            response = self.client.post(
                "/chat/",
                json={"question": "  What is the policy?  "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["question"], "What is the policy?")
        answer_question.assert_called_once_with(
            "What is the policy?",
            3,
            user_id=AUTHENTICATED_USER.user_id,
            retrieval_mode="keyword",
            min_score=None,
            request_id=ANY,
        )

    def test_chat_requires_valid_authentication_before_rag(self):
        app.dependency_overrides.pop(get_current_user, None)
        with patch("app.routers.chat.answer_question") as answer:
            missing = self.client.post(
                "/chat/",
                json={"question": "What is the policy?"},
            )
            invalid = self.client.post(
                "/chat/",
                headers={"Authorization": "Bearer invalid-token"},
                json={"question": "What is the policy?"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        answer.assert_not_called()

    def test_chat_permission_database_failure_returns_safe_503(self):
        with patch(
            "app.services.rag_service.get_readable_document_ids",
            side_effect=AccessControlUnavailableError("private database detail"),
        ), patch("app.services.rag_service.search_chunks") as search:
            response = self.client.post(
                "/chat/",
                json={"question": "What is the policy?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Document authorization is temporarily unavailable.",
        )
        self.assertNotIn("private database detail", response.text)
        search.assert_not_called()

    def test_upload_txt_document_returns_preview_and_chunk_count(self):
        fake_chunks = [
            {
                "chunk_id": "doc-1",
                "document_id": "doc",
                "filename": "policy.txt",
                "position": 1,
                "content": "Annual leave policy",
            }
        ]

        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch("app.routers.documents.index_document", return_value=fake_chunks):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Annual leave policy", "text/plain")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["message"], "document uploaded successfully")
        self.assertEqual(payload["data"]["filename"], "policy.txt")
        self.assertEqual(payload["data"]["chunk_count"], 1)
        self.assertEqual(payload["data"]["preview"], "Annual leave policy")
        self.assertTrue(Path(payload["data"]["saved_path"]).exists())
        self.assertTrue(Path(payload["data"]["text_path"]).exists())
        registration = self.registry_write.call_args.args[0]
        self.assertEqual(registration.original_filename, "policy.txt")
        self.assertEqual(registration.upload_path.split("/")[0], "uploads")
        self.assertEqual(registration.text_path.split("/")[0], "texts")
        self.assertEqual(registration.chunk_count, 1)
        self.assertEqual(registration.owner_id, AUTHENTICATED_USER.user_id)

    def test_upload_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/documents/upload",
            files={"file": ("policy.exe", b"not allowed", "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_upload_rejects_empty_file(self):
        response = self.client.post(
            "/documents/upload",
            files={"file": ("policy.txt", b"", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Uploaded file is empty.")

    def test_upload_rejects_missing_filename(self):
        response = self.client.post(
            "/documents/upload",
            files={"file": ("", b"Policy text", "text/plain")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "file"])

    def test_upload_save_failure_returns_generic_error(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch(
            "app.routers.documents.Path.write_bytes",
            side_effect=OSError("private filesystem detail"),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy text", "text/plain")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Failed to save uploaded file.")
        self.assertNotIn("private filesystem detail", response.text)

    def test_upload_extraction_failure_returns_generic_error(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch(
            "app.routers.documents.extract_document",
            side_effect=RuntimeError("private parser detail"),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy text", "text/plain")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Failed to extract text from document.",
        )
        self.assertNotIn("private parser detail", response.text)

    def test_upload_rejects_whitespace_only_document(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("empty.txt", b" \n\t ", "text/plain")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Document does not contain extractable text.",
        )

    def test_upload_rejects_malformed_pdf(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Document could not be parsed.")

    def test_upload_assigns_unique_document_ids(self):
        fake_chunks = [{"chunk_id": "chunk"}]

        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch("app.routers.documents.index_document", return_value=fake_chunks):
            first = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )
            second = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(
            first.json()["data"]["document_id"],
            second.json()["data"]["document_id"],
        )

    def test_upload_ignores_client_owner_spoofing(self):
        fake_chunks = [{"chunk_id": "chunk"}]
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch("app.routers.documents.index_document", return_value=fake_chunks):
            response = self.client.post(
                "/documents/upload",
                data={"owner_id": str(UUID(int=999))},
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        registration = self.registry_write.call_args.args[0]
        self.assertEqual(registration.owner_id, AUTHENTICATED_USER.user_id)

    def test_upload_embedding_failure_returns_safe_503(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch(
            "app.routers.documents.index_document",
            side_effect=EmbeddingClientError("private embedding provider detail"),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Document vector indexing is temporarily unavailable.",
        )
        self.assertNotIn("private embedding provider detail", response.text)
        self.registry_write.assert_not_called()

    def test_upload_stops_before_file_write_when_registry_is_unavailable(self):
        with patch(
            "app.routers.documents.verify_document_registry_available",
            side_effect=DocumentRegistryUnavailableError("private database detail"),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Document metadata storage is temporarily unavailable.",
        )
        self.assertEqual(list(self.upload_dir.iterdir()), [])
        self.assertNotIn("private database detail", response.text)

    def test_upload_reports_final_metadata_failure_without_success(self):
        fake_chunks = [{"chunk_id": "chunk"}]
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ), patch(
            "app.routers.documents.index_document",
            return_value=fake_chunks,
        ), patch(
            "app.routers.documents.register_document",
            side_effect=DocumentRegistryUnavailableError("private database detail"),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("policy.txt", b"Policy", "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Document was indexed but its metadata could not be persisted.",
        )
        self.assertNotIn("private database detail", response.text)

    def test_documents_list_response(self):
        documents = [
            {
                "document_id": "doc",
                "filename": "policy.md",
                "chunk_count": 2,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]

        with patch(
            "app.routers.documents.list_registered_documents",
            return_value=documents,
        ), patch(
            "app.services.knowledge_base.list_documents",
            side_effect=AssertionError("chunks.json listing fallback was used"),
        ):
            response = self.client.get("/documents/")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["documents"][0]["document_id"], "doc")
        self.assertEqual(payload["data"]["documents"][0]["chunk_count"], 2)
        self.document_list_authorization.assert_called_once_with(
            AUTHENTICATED_USER.user_id
        )

    def test_documents_list_returns_safe_503_when_registry_is_unavailable(self):
        with patch(
            "app.routers.documents.list_registered_documents",
            side_effect=DocumentRegistryUnavailableError("private database detail"),
        ):
            response = self.client.get("/documents/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Document metadata storage is temporarily unavailable.",
        )
        self.assertNotIn("private database detail", response.text)

    def test_document_preview_reads_extracted_text(self):
        document_id = "doc-preview"
        (self.text_dir / f"{document_id}.txt").write_text("Preview text", encoding="utf-8")

        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ):
            response = self.client.get(f"/documents/{document_id}/preview")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["document_id"], document_id)
        self.assertEqual(payload["data"]["preview"], "Preview text")

    def test_document_preview_returns_404_when_text_is_missing(self):
        with patch(
            "app.routers.documents.get_document_storage_paths",
            return_value=self.storage_paths,
        ):
            response = self.client.get("/documents/missing/preview")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Document text not found", response.json()["detail"])

    def test_document_chunks_response(self):
        chunks = [
            {
                "chunk_id": "doc-1",
                "document_id": "doc",
                "filename": "policy.md",
                "position": 1,
                "content": "Chunk text",
                "token_count": 2,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]

        with patch("app.routers.documents.get_document_chunks", return_value=chunks):
            response = self.client.get("/documents/doc/chunks")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["chunk_count"], 1)
        self.assertEqual(payload["data"]["chunks"][0]["chunk_id"], "doc-1")

    def test_document_chunks_exposes_page_metadata(self):
        chunks = [
            {
                "chunk_id": "doc-1",
                "document_id": "doc",
                "filename": "policy.pdf",
                "position": 1,
                "chunk_index": 0,
                "page_number": 7,
                "content": "Page-aware policy text",
                "token_count": 3,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]

        with patch("app.routers.documents.get_document_chunks", return_value=chunks):
            response = self.client.get("/documents/doc/chunks")

        payload = response.json()["data"]["chunks"][0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["chunk_index"], 0)
        self.assertEqual(payload["page_number"], 7)

    def test_document_chunks_returns_404_when_chunks_are_missing(self):
        with patch("app.routers.documents.get_document_chunks", return_value=[]):
            response = self.client.get("/documents/missing/chunks")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Document chunks not found", response.json()["detail"])

    def test_chat_no_context_response(self):
        result = {
            "question": "What is the policy?",
            "retrieval_mode": "keyword",
            "min_score": 0.2,
            "answer": "No reliable answer.",
            "answer_mode": "no_context",
            "model": None,
            "llm_error": None,
            "citations": [],
            "contexts": [],
        }
        with patch("app.routers.chat.answer_question", return_value=result):
            response = self.client.post(
                "/chat/",
                json={"question": "What is the policy?", "top_k": 2},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["answer_mode"], "no_context")
        self.assertEqual(payload["data"]["contexts"], [])
        self.assertEqual(payload["data"]["citations"], [])

    def test_chat_uses_storage_path_provider_for_default_index(self):
        chunks_file = self.storage_paths.chunks_file
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        chunks_file.write_text(
            """[
  {
    "chunk_id": "doc-1",
    "document_id": "doc",
    "filename": "policy.md",
    "position": 1,
    "content": "Remote work requires manager approval.",
    "token_count": 5,
    "created_at": "2026-01-01T00:00:00+00:00"
  }
]""",
            encoding="utf-8",
        )

        with patch(
            "app.services.knowledge_base.get_document_storage_paths",
            return_value=self.storage_paths,
        ):
            response = self.client.post(
                "/chat/",
                json={
                    "question": "remote manager approval",
                    "retrieval_mode": "keyword",
                    "min_score": 0.0,
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["answer_mode"], "local_fallback")
        self.assertEqual(payload["data"]["contexts"][0]["chunk_id"], "doc-1")
        self.assertEqual(payload["data"]["answer"], SAFE_BLOCKED_RESPONSE)
        self.assertEqual(payload["data"]["citations"], [])

    def test_chat_with_context_returns_citations(self):
        contexts = [
            {
                "chunk_id": "doc-1",
                "document_id": "doc",
                "filename": "policy.md",
                "position": 1,
                "chunk_index": 0,
                "page_number": 7,
                "content": "Policy text",
                "score": 0.9,
                "retrieval_mode": "keyword",
                "context_role": "retrieved",
            }
        ]
        result = {
            "question": "What is the policy?",
            "retrieval_mode": "keyword",
            "min_score": 0.2,
            "answer": "Use the policy.",
            "answer_mode": "local_fallback",
            "model": None,
            "llm_error": "LLM API key is not configured.",
            "citations": [
                {
                    "chunk_id": "doc-1",
                    "document_id": "doc",
                    "filename": "policy.md",
                    "position": 1,
                    "chunk_index": 0,
                    "page_number": 7,
                    "score": 0.9,
                    "retrieval_mode": "keyword",
                    "context_role": "retrieved",
                    "expanded_from_chunk_id": None,
                }
            ],
            "contexts": contexts,
        }

        with patch("app.routers.chat.answer_question", return_value=result):
            response = self.client.post(
                "/chat/",
                json={"question": "What is the policy?", "retrieval_mode": "keyword"},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["answer"], "Use the policy.")
        self.assertEqual(payload["data"]["citations"][0]["chunk_id"], "doc-1")
        self.assertEqual(payload["data"]["citations"][0]["page_number"], 7)
        self.assertEqual(payload["data"]["contexts"][0]["chunk_id"], "doc-1")

    def test_chat_returns_llm_answer_when_provider_succeeds(self):
        contexts = [
            {
                "chunk_id": "doc-1",
                "document_id": "doc",
                "filename": "policy.md",
                "position": 1,
                "content": "Managers approve annual leave.",
                "score": 0.9,
                "retrieval_mode": "keyword",
                "context_role": "retrieved",
            }
        ]

        with patch("app.services.rag_service.search_chunks", return_value=contexts), patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch(
            "app.services.knowledge_base.chat_completion",
            return_value="Annual leave requires manager approval.",
        ) as completion:
            response = self.client.post(
                "/chat/",
                json={"question": "How is annual leave approved?"},
            )

        payload = response.json()["data"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["answer_mode"], "llm")
        self.assertEqual(payload["answer"], "Annual leave requires manager approval.")
        self.assertIsNone(payload["llm_error"])
        messages = completion.call_args.args[0]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Managers approve annual leave.", messages[1]["content"])

    def test_chat_provider_failure_returns_only_safe_error(self):
        contexts = [
            {
                "chunk_id": "doc-1",
                "document_id": "doc",
                "filename": "policy.md",
                "position": 1,
                "content": "Policy text",
                "score": 0.9,
                "retrieval_mode": "keyword",
                "context_role": "retrieved",
            }
        ]
        error = LLMClientError(
            "private provider diagnostic",
            code="provider_unavailable",
            public_message="LLM provider is temporarily unavailable.",
            retryable=True,
        )

        with patch("app.services.rag_service.search_chunks", return_value=contexts), patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch(
            "app.services.knowledge_base.chat_completion",
            side_effect=error,
        ):
            response = self.client.post(
                "/chat/",
                json={"question": "What is the policy?"},
            )

        payload = response.json()["data"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["answer_mode"], "local_fallback")
        self.assertEqual(
            payload["llm_error"],
            "LLM provider is temporarily unavailable.",
        )
        self.assertNotIn("private provider diagnostic", response.text)


if __name__ == "__main__":
    unittest.main()
