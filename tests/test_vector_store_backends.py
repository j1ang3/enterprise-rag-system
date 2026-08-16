import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.services import vector_store


CHUNKS = [
    {
        "chunk_id": "doc-1",
        "document_id": "doc",
        "filename": "doc.md",
        "position": 1,
        "content": "alpha",
        "created_at": "2026-01-01T00:00:00+00:00",
    },
    {
        "chunk_id": "doc-2",
        "document_id": "doc",
        "filename": "doc.md",
        "position": 2,
        "content": "beta",
        "created_at": "2026-01-01T00:00:00+00:00",
    },
]


def dense_embedding(text):
    if text == "alpha" or text == "query alpha":
        return [1.0, 0.0]
    if text == "beta":
        return [0.0, 1.0]
    return [0.0, 0.0]


def sparse_embedding(text):
    if text == "alpha" or text == "query alpha":
        return {"0": 1.0}
    if text == "beta":
        return {"1": 1.0}
    return {}


class VectorStoreBackendTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = settings.vector_store_backend
        self.previous_external_provider = settings.vector_store_external_provider
        self.previous_vector_store_url = settings.vector_store_url
        self.previous_vector_store_collection = settings.vector_store_collection
        settings.vector_store_backend = "auto"
        settings.vector_store_external_provider = ""
        settings.vector_store_url = ""
        settings.vector_store_collection = "enterprise_rag_chunks"
        self.tempdir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.tempdir.name) / "vectors.json"

    def tearDown(self):
        settings.vector_store_backend = self.previous_backend
        settings.vector_store_external_provider = self.previous_external_provider
        settings.vector_store_url = self.previous_vector_store_url
        settings.vector_store_collection = self.previous_vector_store_collection
        self.tempdir.cleanup()

    def test_auto_uses_faiss_when_index_is_available(self):
        if vector_store._load_faiss_module() is None:
            self.skipTest("faiss is not installed")

        with patch.object(vector_store, "embed_text", side_effect=dense_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="test-model",
        ):
            vector_store.rebuild_vector_index(CHUNKS, self.index_path)
            results = vector_store.search_vector_chunks(
                "query alpha",
                top_k=1,
                index_path=self.index_path,
                allowed_document_ids={"doc"},
            )

        self.assertTrue(vector_store._faiss_index_path(self.index_path).exists())
        self.assertTrue(vector_store._faiss_metadata_path(self.index_path).exists())
        self.assertEqual(results[0]["chunk_id"], "doc-1")
        self.assertEqual(results[0]["vector_store_backend"], "faiss")

    def test_auto_falls_back_to_numpy_when_faiss_is_unavailable(self):
        with patch.object(vector_store, "_load_faiss_module", return_value=None), patch.object(
            vector_store,
            "embed_text",
            side_effect=dense_embedding,
        ), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="test-model",
        ):
            vector_store.rebuild_vector_index(CHUNKS, self.index_path)
            results = vector_store.search_vector_chunks(
                "query alpha",
                top_k=1,
                index_path=self.index_path,
                allowed_document_ids={"doc"},
            )

        self.assertTrue(vector_store._matrix_index_path(self.index_path).exists())
        self.assertTrue(vector_store._matrix_metadata_path(self.index_path).exists())
        self.assertFalse(vector_store._faiss_index_path(self.index_path).exists())
        self.assertEqual(results[0]["chunk_id"], "doc-1")
        self.assertEqual(results[0]["vector_store_backend"], "numpy")

    def test_sparse_embeddings_fall_back_to_json_scanning(self):
        with patch.object(vector_store, "embed_text", side_effect=sparse_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="local-hashed-v1",
        ):
            vector_store.rebuild_vector_index(CHUNKS, self.index_path)
            results = vector_store.search_vector_chunks(
                "query alpha",
                top_k=1,
                index_path=self.index_path,
                allowed_document_ids={"doc"},
            )

        self.assertFalse(vector_store._matrix_index_path(self.index_path).exists())
        self.assertFalse(vector_store._faiss_index_path(self.index_path).exists())
        self.assertEqual(results[0]["chunk_id"], "doc-1")
        self.assertEqual(results[0]["vector_store_backend"], "json")

    def test_auto_backend_order_prefers_faiss_then_numpy_then_json(self):
        backend_names = [
            backend.name
            for backend in vector_store.get_vector_search_backends("auto")
        ]

        self.assertEqual(backend_names, ["faiss", "numpy", "json"])

    def test_explicit_json_backend_uses_json_only(self):
        backend_names = [
            backend.name
            for backend in vector_store.get_vector_search_backends("json")
        ]

        self.assertEqual(backend_names, ["json"])

    def test_external_backend_order_keeps_local_fallbacks(self):
        backend_names = [
            backend.name
            for backend in vector_store.get_vector_search_backends("external")
        ]

        self.assertEqual(backend_names, ["external", "faiss", "numpy", "json"])

    def test_external_index_backend_order_keeps_local_fallback(self):
        backend_names = [
            backend.name
            for backend in vector_store.get_vector_index_backends("external")
        ]

        self.assertEqual(backend_names, ["external", "local"])

    def test_unconfigured_external_backend_falls_back_to_local_json(self):
        settings.vector_store_backend = "external"

        with patch.object(vector_store, "embed_text", side_effect=sparse_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="local-hashed-v1",
        ):
            vector_store.rebuild_vector_index(CHUNKS, self.index_path)
            results = vector_store.search_vector_chunks(
                "query alpha",
                top_k=1,
                index_path=self.index_path,
                allowed_document_ids={"doc"},
            )

        self.assertEqual(results[0]["chunk_id"], "doc-1")
        self.assertEqual(results[0]["vector_store_backend"], "json")

    def test_local_index_upsert_replaces_existing_document_entries(self):
        with patch.object(vector_store, "embed_text", side_effect=dense_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="test-model",
        ):
            vector_store.rebuild_vector_index(CHUNKS, self.index_path)
            vector_store.index_vector_chunks("doc", [CHUNKS[0]], self.index_path)

        entries = vector_store._load_vectors(self.index_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["chunk_id"], "doc-1")

    def test_external_configuration_detection(self):
        self.assertFalse(vector_store._is_external_vector_store_configured())

        settings.vector_store_external_provider = "qdrant"
        settings.vector_store_url = "http://localhost:6333"

        self.assertTrue(vector_store._is_external_vector_store_configured())

    def test_configured_qdrant_search_returns_qdrant_results(self):
        settings.vector_store_backend = "external"
        settings.vector_store_external_provider = "qdrant"
        settings.vector_store_url = "http://localhost:6333"

        qdrant_response = {
            "result": [
                {
                    "score": 0.91,
                    "payload": {
                        "chunk_id": "doc-1",
                        "document_id": "doc",
                        "filename": "doc.md",
                        "position": 1,
                        "content": "alpha",
                        "embedding_model": "test-model",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    },
                }
            ]
        }

        with patch.object(vector_store, "embed_text", side_effect=dense_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="test-model",
        ), patch.object(vector_store, "_qdrant_request", return_value=qdrant_response) as request:
            results = vector_store.search_vector_chunks(
                "query alpha",
                top_k=1,
                index_path=self.index_path,
                allowed_document_ids={"doc"},
            )

        self.assertEqual(results[0]["chunk_id"], "doc-1")
        self.assertEqual(results[0]["vector_store_backend"], "qdrant")
        request.assert_called_once()
        self.assertEqual(
            request.call_args.args[2]["filter"],
            {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"any": ["doc"]},
                    }
                ]
            },
        )

    def test_empty_permission_set_stops_before_embedding(self):
        with patch.object(vector_store, "embed_text") as embed:
            results = vector_store.search_vector_chunks(
                "query alpha",
                top_k=3,
                index_path=self.index_path,
                allowed_document_ids=set(),
            )

        self.assertEqual(results, [])
        embed.assert_not_called()

    def test_faiss_adaptive_overfetch_recovers_lower_ranked_authorized_results(self):
        if vector_store._load_faiss_module() is None:
            self.skipTest("faiss is not installed")

        entries = []
        for index in range(8):
            entries.append(
                {
                    "chunk_id": f"blocked-{index}",
                    "document_id": "blocked-doc",
                    "filename": "blocked.md",
                    "position": index + 1,
                    "content": "blocked",
                    "embedding": [1.0 - index * 0.01, 0.0],
                    "embedding_model": "test-model",
                }
            )
        entries.extend(
            [
                {
                    "chunk_id": "allowed-1",
                    "document_id": "allowed-doc",
                    "filename": "allowed.md",
                    "position": 1,
                    "content": "allowed one",
                    "embedding": [0.50, 0.0],
                    "embedding_model": "test-model",
                },
                {
                    "chunk_id": "allowed-2",
                    "document_id": "allowed-doc",
                    "filename": "allowed.md",
                    "position": 2,
                    "content": "allowed two",
                    "embedding": [0.40, 0.0],
                    "embedding_model": "test-model",
                },
            ]
        )
        vector_store._save_vectors(entries, self.index_path)

        with patch.object(vector_store, "embed_text", return_value=[1.0, 0.0]), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="test-model",
        ):
            results = vector_store.search_vector_chunks(
                "query",
                top_k=2,
                index_path=self.index_path,
                allowed_document_ids={"allowed-doc"},
            )

        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["allowed-1", "allowed-2"],
        )
        self.assertTrue(all(result["document_id"] == "allowed-doc" for result in results))

    def test_configured_qdrant_upsert_writes_points(self):
        settings.vector_store_backend = "external"
        settings.vector_store_external_provider = "qdrant"
        settings.vector_store_url = "http://localhost:6333"

        with patch.object(vector_store, "embed_text", side_effect=dense_embedding), patch.object(
            vector_store,
            "embedding_model_for_vector",
            return_value="test-model",
        ), patch.object(vector_store, "_qdrant_request", return_value={}) as request:
            entries = vector_store.index_vector_chunks("doc", [CHUNKS[0]], self.index_path)

        self.assertEqual(entries[0]["chunk_id"], "doc-1")
        self.assertEqual(request.call_count, 3)
        methods_and_paths = [
            (call.args[0], call.args[1])
            for call in request.call_args_list
        ]
        self.assertEqual(methods_and_paths[0], ("GET", "/collections/enterprise_rag_chunks"))
        self.assertEqual(
            methods_and_paths[1],
            ("POST", "/collections/enterprise_rag_chunks/points/delete?wait=true"),
        )
        self.assertEqual(
            methods_and_paths[2],
            ("PUT", "/collections/enterprise_rag_chunks/points?wait=true"),
        )


if __name__ == "__main__":
    unittest.main()
