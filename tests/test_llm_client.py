import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from app.core.config import settings
from app.services.llm_client import (
    ChatCompletionResult,
    LLMClientError,
    chat_completion,
    get_llm_runtime_metadata,
)


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class LLMClientTests(unittest.TestCase):
    def test_metadata_result_preserves_actual_model_and_usage(self):
        response = FakeResponse(
            json.dumps(
                {
                    "model": "qwen3:8b",
                    "choices": [{"message": {"content": "Answer"}}],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                }
            ).encode("utf-8")
        )
        with patch.object(settings, "llm_api_key", "test-key"), patch(
            "app.services.llm_client.urlopen", return_value=response
        ):
            result = chat_completion(
                [{"role": "user", "content": "Hello"}], include_metadata=True
            )

        self.assertIsInstance(result, ChatCompletionResult)
        self.assertEqual(result.model, "qwen3:8b")
        self.assertEqual(result.total_tokens, 16)

    def test_metadata_result_uses_null_when_provider_omits_usage(self):
        response = FakeResponse(
            b'{"model":"gemma3:4b","choices":[{"message":{"content":"Answer"}}]}'
        )
        with patch.object(settings, "llm_api_key", "test-key"), patch(
            "app.services.llm_client.urlopen", return_value=response
        ):
            result = chat_completion(
                [{"role": "user", "content": "Hello"}], include_metadata=True
            )

        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.total_tokens)

    def test_ollama_runtime_metadata_resolves_model_identity(self):
        tags = FakeResponse(
            json.dumps(
                {
                    "models": [
                        {
                            "name": "qwen3:8b",
                            "digest": "digest-123",
                            "modified_at": "2026-08-05T00:00:00Z",
                            "details": {"parameter_size": "8B"},
                        }
                    ]
                }
            ).encode("utf-8")
        )
        version = FakeResponse(b'{"version":"0.32.5"}')

        with patch.object(settings, "llm_provider", "ollama"), patch.object(
            settings, "llm_base_url", "http://127.0.0.1:11434/v1/"
        ), patch.object(settings, "llm_model", "qwen3:8b"), patch(
            "app.services.llm_client.urlopen",
            side_effect=[tags, version],
        ) as opener:
            metadata = get_llm_runtime_metadata(resolve_model_identity=True)

        self.assertEqual(metadata["provider"], "ollama")
        self.assertEqual(metadata["model"], "qwen3:8b")
        self.assertEqual(metadata["ollama_version"], "0.32.5")
        self.assertEqual(metadata["model_identity"]["digest"], "digest-123")
        self.assertEqual(
            opener.call_args_list[0].args[0].full_url,
            "http://127.0.0.1:11434/api/tags",
        )

    def test_ollama_runtime_metadata_rejects_missing_model(self):
        with patch.object(settings, "llm_provider", "ollama"), patch.object(
            settings, "llm_base_url", "http://127.0.0.1:11434/v1"
        ), patch.object(settings, "llm_model", "missing:latest"), patch(
            "app.services.llm_client.urlopen",
            return_value=FakeResponse(b'{"models":[]}'),
        ):
            with self.assertRaises(LLMClientError) as raised:
                get_llm_runtime_metadata(resolve_model_identity=True)

        self.assertEqual(raised.exception.code, "model_not_available")

    def test_requires_api_key(self):
        with patch.object(settings, "llm_api_key", ""):
            with self.assertRaises(LLMClientError) as raised:
                chat_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.code, "not_configured")
        self.assertEqual(
            raised.exception.public_message,
            "LLM service is not configured.",
        )

    def test_success_sends_openai_compatible_payload(self):
        response = FakeResponse(
            json.dumps(
                {"choices": [{"message": {"content": "  Model answer  "}}]}
            ).encode("utf-8")
        )
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(settings, "llm_api_key", "test-key"), patch.object(
            settings, "llm_base_url", "https://provider.example/v1/"
        ), patch.object(settings, "llm_model", "test-model"), patch.object(
            settings, "llm_temperature", 0.3
        ), patch.object(settings, "llm_max_tokens", 128), patch.object(
            settings, "llm_seed", None
        ), patch.object(
            settings, "llm_reasoning_effort", ""
        ), patch.object(
            settings, "llm_timeout_seconds", 4.0
        ), patch("app.services.llm_client.urlopen", return_value=response) as opener:
            answer = chat_completion(messages)

        self.assertEqual(answer, "Model answer")
        request = opener.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://provider.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(opener.call_args.kwargs["timeout"], 4.0)
        self.assertEqual(
            payload,
            {
                "model": "test-model",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 128,
            },
        )

    def test_optional_seed_is_sent_when_configured(self):
        response = FakeResponse(
            b'{"choices":[{"message":{"content":"Seeded answer"}}]}'
        )

        with patch.object(settings, "llm_api_key", "test-key"), patch.object(
            settings, "llm_seed", 42
        ), patch(
            "app.services.llm_client.urlopen",
            return_value=response,
        ) as opener:
            answer = chat_completion([{"role": "user", "content": "Hello"}])

        payload = json.loads(opener.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(answer, "Seeded answer")
        self.assertEqual(payload["seed"], 42)

    def test_optional_reasoning_effort_is_sent_when_configured(self):
        response = FakeResponse(
            b'{"choices":[{"message":{"content":"Direct answer"}}]}'
        )

        with patch.object(settings, "llm_api_key", "test-key"), patch.object(
            settings, "llm_reasoning_effort", "none"
        ), patch(
            "app.services.llm_client.urlopen",
            return_value=response,
        ) as opener:
            answer = chat_completion([{"role": "user", "content": "Hello"}])

        payload = json.loads(opener.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(answer, "Direct answer")
        self.assertEqual(payload["reasoning_effort"], "none")

    def test_authentication_error_is_not_retried_or_leaked(self):
        error = HTTPError(
            "https://provider.example/v1/chat/completions",
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"internal":"private provider detail"}'),
        )

        with patch.object(settings, "llm_api_key", "invalid-key"), patch(
            "app.services.llm_client.urlopen",
            side_effect=error,
        ) as opener:
            with self.assertRaises(LLMClientError) as raised:
                chat_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(opener.call_count, 1)
        self.assertEqual(raised.exception.code, "authentication_error")
        self.assertEqual(
            raised.exception.public_message,
            "LLM provider authentication failed.",
        )
        self.assertNotIn("private provider detail", str(raised.exception))

    def test_retryable_provider_error_retries_then_succeeds(self):
        error = HTTPError(
            "https://provider.example/v1/chat/completions",
            503,
            "Unavailable",
            hdrs=None,
            fp=None,
        )
        response = FakeResponse(
            b'{"choices":[{"message":{"content":"Recovered"}}]}'
        )

        with patch.object(settings, "llm_api_key", "test-key"), patch.object(
            settings, "llm_max_retries", 1
        ), patch.object(settings, "llm_retry_backoff_seconds", 0.25), patch(
            "app.services.llm_client.urlopen",
            side_effect=[error, response],
        ) as opener, patch("app.services.llm_client.time.sleep") as sleep:
            answer = chat_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(answer, "Recovered")
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_timeout_retries_then_returns_safe_error(self):
        with patch.object(settings, "llm_api_key", "test-key"), patch.object(
            settings, "llm_max_retries", 1
        ), patch.object(settings, "llm_retry_backoff_seconds", 0.0), patch(
            "app.services.llm_client.urlopen",
            side_effect=[TimeoutError("private timeout detail"), TimeoutError("again")],
        ) as opener:
            with self.assertRaises(LLMClientError) as raised:
                chat_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(opener.call_count, 2)
        self.assertEqual(raised.exception.code, "timeout")
        self.assertEqual(
            raised.exception.public_message,
            "LLM provider request timed out.",
        )
        self.assertNotIn("private timeout detail", raised.exception.public_message)

    def test_network_failure_returns_provider_unavailable(self):
        with patch.object(settings, "llm_api_key", "test-key"), patch.object(
            settings, "llm_max_retries", 0
        ), patch(
            "app.services.llm_client.urlopen",
            side_effect=URLError("private DNS detail"),
        ):
            with self.assertRaises(LLMClientError) as raised:
                chat_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.code, "provider_unavailable")
        self.assertEqual(
            raised.exception.public_message,
            "LLM provider is temporarily unavailable.",
        )
        self.assertNotIn("private DNS detail", raised.exception.public_message)

    def test_invalid_json_returns_invalid_response_error(self):
        with patch.object(settings, "llm_api_key", "test-key"), patch(
            "app.services.llm_client.urlopen",
            return_value=FakeResponse(b"not-json"),
        ):
            with self.assertRaises(LLMClientError) as raised:
                chat_completion([{"role": "user", "content": "Hello"}])

        self.assertEqual(raised.exception.code, "invalid_response")
        self.assertEqual(
            raised.exception.public_message,
            "LLM provider returned an invalid response.",
        )

    def test_missing_or_empty_answer_returns_invalid_response_error(self):
        invalid_responses = [
            b'{"choices":[]}',
            b'{"choices":[{"message":{"content":"   "}}]}',
        ]

        for response_body in invalid_responses:
            with self.subTest(response_body=response_body), patch.object(
                settings, "llm_api_key", "test-key"
            ), patch(
                "app.services.llm_client.urlopen",
                return_value=FakeResponse(response_body),
            ):
                with self.assertRaises(LLMClientError) as raised:
                    chat_completion([{"role": "user", "content": "Hello"}])

                self.assertEqual(raised.exception.code, "invalid_response")


if __name__ == "__main__":
    unittest.main()
