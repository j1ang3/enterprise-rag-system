import json
import time
from dataclasses import dataclass
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from app.core.config import settings


class LLMClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        public_message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = public_message
        self.retryable = retryable


@dataclass(frozen=True)
class ChatCompletionResult:
    """Answer plus provider metadata that is safe to propagate internally."""

    answer: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def token_usage(self) -> Dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def is_llm_configured() -> bool:
    return bool(settings.llm_api_key)


def _ollama_root_url() -> str:
    parts = urlsplit(settings.llm_base_url.rstrip("/"))
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


def _read_json_url(url: str) -> Dict:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            return _decode_response(response.read())
    except (HTTPError, TimeoutError, URLError) as exc:
        raise LLMClientError(
            f"Failed to read LLM runtime metadata from {url}.",
            code="provider_unavailable",
            public_message="LLM provider metadata is unavailable.",
        ) from exc


def get_llm_runtime_metadata(*, resolve_model_identity: bool = False) -> Dict:
    """Describe the configured generation runtime without exposing credentials."""
    metadata: Dict = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "seed": settings.llm_seed,
        "reasoning_effort": settings.llm_reasoning_effort or None,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if not resolve_model_identity or settings.llm_provider.lower() != "ollama":
        return metadata

    ollama_root = _ollama_root_url()
    tags = _read_json_url(f"{ollama_root}/api/tags")
    models = tags.get("models")
    if not isinstance(models, list):
        raise LLMClientError(
            "Ollama model list did not contain a models array.",
            code="invalid_response",
            public_message="LLM provider returned an invalid model list.",
        )

    model_identity = next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and settings.llm_model in {item.get("name"), item.get("model")}
        ),
        None,
    )
    if model_identity is None:
        raise LLMClientError(
            f"Configured Ollama model is unavailable: {settings.llm_model}",
            code="model_not_available",
            public_message="Configured LLM model is unavailable.",
        )

    version = _read_json_url(f"{ollama_root}/api/version")
    metadata["ollama_version"] = version.get("version")
    metadata["model_identity"] = {
        "tag": model_identity.get("name") or model_identity.get("model"),
        "digest": model_identity.get("digest"),
        "modified_at": model_identity.get("modified_at"),
        "details": model_identity.get("details"),
    }
    return metadata


def _http_error_to_client_error(exc: HTTPError) -> LLMClientError:
    if exc.code in {401, 403}:
        return LLMClientError(
            f"LLM provider authentication failed with HTTP {exc.code}.",
            code="authentication_error",
            public_message="LLM provider authentication failed.",
        )
    if exc.code in {408, 504}:
        return LLMClientError(
            f"LLM provider timed out with HTTP {exc.code}.",
            code="timeout",
            public_message="LLM provider request timed out.",
            retryable=True,
        )
    if exc.code == 429:
        return LLMClientError(
            "LLM provider rate limit exceeded.",
            code="rate_limit",
            public_message="LLM provider is temporarily rate limited.",
            retryable=True,
        )
    if 500 <= exc.code <= 599:
        return LLMClientError(
            f"LLM provider failed with HTTP {exc.code}.",
            code="provider_unavailable",
            public_message="LLM provider is temporarily unavailable.",
            retryable=True,
        )
    return LLMClientError(
        f"LLM provider rejected the request with HTTP {exc.code}.",
        code="provider_rejected_request",
        public_message="LLM provider rejected the request.",
    )


def _decode_response(raw_body: bytes) -> Dict:
    try:
        response_data = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMClientError(
            "LLM provider returned a response that was not valid JSON.",
            code="invalid_response",
            public_message="LLM provider returned an invalid response.",
        ) from exc

    if not isinstance(response_data, dict):
        raise LLMClientError(
            "LLM provider returned a non-object JSON response.",
            code="invalid_response",
            public_message="LLM provider returned an invalid response.",
        )
    return response_data


def _extract_answer(response_data: Dict) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError(
            "LLM provider response did not include a chat answer.",
            code="invalid_response",
            public_message="LLM provider returned an invalid response.",
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMClientError(
            "LLM provider response included an empty chat answer.",
            code="invalid_response",
            public_message="LLM provider returned an invalid response.",
        )
    return content.strip()


def _optional_token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _extract_completion_result(response_data: Dict) -> ChatCompletionResult:
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    actual_model = response_data.get("model")
    if not isinstance(actual_model, str) or not actual_model.strip():
        actual_model = settings.llm_model
    return ChatCompletionResult(
        answer=_extract_answer(response_data),
        model=actual_model.strip(),
        prompt_tokens=_optional_token_count(usage.get("prompt_tokens")),
        completion_tokens=_optional_token_count(usage.get("completion_tokens")),
        total_tokens=_optional_token_count(usage.get("total_tokens")),
    )


def _sleep_before_retry(attempt: int) -> None:
    delay = settings.llm_retry_backoff_seconds * (2 ** attempt)
    if delay > 0:
        time.sleep(delay)


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    include_metadata: bool = False,
) -> str | ChatCompletionResult:
    """
    Call an OpenAI-compatible chat completions API.
    """
    if not is_llm_configured():
        raise LLMClientError(
            "LLM API key is not configured.",
            code="not_configured",
            public_message="LLM service is not configured.",
        )

    base_url = settings.llm_base_url.rstrip("/")
    request_url = f"{base_url}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    if settings.llm_seed is not None:
        payload["seed"] = settings.llm_seed
    if settings.llm_reasoning_effort:
        payload["reasoning_effort"] = settings.llm_reasoning_effort
    body = json.dumps(payload).encode("utf-8")

    request = Request(
        request_url,
        data=body,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(settings.llm_max_retries + 1):
        try:
            with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
                response_data = _decode_response(response.read())
                completion = _extract_completion_result(response_data)
                return completion if include_metadata else completion.answer
        except HTTPError as exc:
            client_error = _http_error_to_client_error(exc)
        except TimeoutError as exc:
            client_error = LLMClientError(
                "LLM provider request timed out.",
                code="timeout",
                public_message="LLM provider request timed out.",
                retryable=True,
            )
            client_error.__cause__ = exc
        except URLError as exc:
            client_error = LLMClientError(
                f"Failed to reach LLM provider: {exc.reason}",
                code="provider_unavailable",
                public_message="LLM provider is temporarily unavailable.",
                retryable=True,
            )
            client_error.__cause__ = exc

        if not client_error.retryable or attempt >= settings.llm_max_retries:
            raise client_error
        _sleep_before_retry(attempt)

    raise AssertionError("LLM retry loop exited unexpectedly.")
