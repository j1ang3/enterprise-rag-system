import json
import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services import embeddings


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.previous_provider = settings.embedding_provider
        self.previous_fallback = settings.embedding_fallback_to_local
        self.previous_api_key = settings.embedding_api_key
        self.previous_base_url = settings.embedding_base_url
        self.previous_model = settings.embedding_model

    def tearDown(self):
        settings.embedding_provider = self.previous_provider
        settings.embedding_fallback_to_local = self.previous_fallback
        settings.embedding_api_key = self.previous_api_key
        settings.embedding_base_url = self.previous_base_url
        settings.embedding_model = self.previous_model

    def test_local_hashed_embedding_is_deterministic_and_normalized(self):
        first = embeddings._embed_text_local("Annual leave policy")
        second = embeddings._embed_text_local("Annual leave policy")

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            sum(value * value for value in first.values()),
            1.0,
            places=5,
        )

    def test_embed_text_dispatches_to_configured_local_hashed_provider(self):
        settings.embedding_provider = "local_hashed"

        vector = embeddings.embed_text("travel approval")

        self.assertIsInstance(vector, dict)
        self.assertEqual(
            embeddings.embedding_model_for_vector(vector),
            embeddings.LOCAL_EMBEDDING_MODEL,
        )

    def test_openai_compatible_provider_sends_embedding_request(self):
        settings.embedding_provider = "openai_compatible"
        settings.embedding_fallback_to_local = False
        settings.embedding_api_key = "test-key"
        settings.embedding_base_url = "https://embedding.example/v1"
        settings.embedding_model = "test-embedding-model"

        with patch(
            "app.services.embeddings.urlopen",
            return_value=FakeResponse({"data": [{"embedding": [0.1, 0.2]}]}),
        ) as urlopen:
            vector = embeddings.embed_text("annual leave")

        self.assertEqual(vector, [0.1, 0.2])
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://embedding.example/v1/embeddings")
        self.assertEqual(payload["model"], "test-embedding-model")
        self.assertEqual(payload["input"], "annual leave")

    def test_unknown_provider_without_fallback_is_explicit_error(self):
        settings.embedding_provider = "unknown"
        settings.embedding_fallback_to_local = False

        with self.assertRaises(embeddings.EmbeddingClientError):
            embeddings.embed_text("policy")


if __name__ == "__main__":
    unittest.main()
