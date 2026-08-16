import hashlib
import json
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import settings


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
DEFAULT_DIMENSIONS = 384
LOCAL_EMBEDDING_MODEL = "local-hashed-v1"
Embedding = Union[Dict[str, float], List[float]]
_LOCAL_MODEL = None
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "is",
    "it",
    "of",
    "the",
    "this",
    "to",
    "what",
    "with",
}
TOKEN_EXPANSIONS = {
    "abroad": ["country", "international", "remote", "work", "travel", "approval", "hr", "security"],
    "applied": ["apply", "effect"],
    "changed": ["change", "permission"],
    "consecutive": ["row"],
    "document": ["certificate"],
    "effect": ["applied"],
    "file": ["submit"],
    "ill": ["sick"],
    "international": ["country", "abroad", "travel"],
    "one": ["1"],
    "permission": ["access"],
    "permissions": ["permission", "access"],
    "report": ["reported"],
    "reported": ["report"],
    "row": ["consecutive"],
    "sick": ["medical", "certificate"],
    "three": ["3"],
    "two": ["2"],
}


def _normalize_token(token: str) -> str:
    normalized = token.lower()
    if len(normalized) > 4 and normalized.endswith("ing"):
        normalized = normalized[:-3]
    elif len(normalized) > 4 and normalized.endswith("ied"):
        normalized = f"{normalized[:-3]}y"
    elif len(normalized) > 4 and normalized.endswith("ed") and not normalized.endswith("ged"):
        normalized = normalized[:-2]
    elif len(normalized) > 3 and normalized.endswith("es"):
        normalized = normalized[:-2]
    elif len(normalized) > 3 and normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized


def tokenize(text: str) -> List[str]:
    tokens = []

    for token in TOKEN_PATTERN.findall(text):
        normalized = _normalize_token(token)

        if normalized not in STOP_WORDS:
            tokens.append(normalized)
            tokens.extend(
                expansion
                for expansion in TOKEN_EXPANSIONS.get(normalized, [])
                if expansion not in STOP_WORDS
            )

    return tokens


def _features(tokens: Iterable[str]) -> Counter:
    features = Counter()
    token_list = list(tokens)

    for token in token_list:
        features[f"tok:{token}"] += 1.0

        if len(token) >= 4 and token.isascii():
            for index in range(0, len(token) - 2):
                features[f"tri:{token[index:index + 3]}"] += 0.35

    for left, right in zip(token_list, token_list[1:]):
        features[f"bi:{left}_{right}"] += 0.75

    return features


def _hash_feature(feature: str, dimensions: int) -> tuple[str, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], byteorder="big") % dimensions
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return str(bucket), sign


class EmbeddingClientError(RuntimeError):
    pass


def is_embedding_configured() -> bool:
    return settings.embedding_provider.lower() == "openai_compatible" and bool(settings.embedding_api_key)


def _embed_text_local(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> Dict[str, float]:
    """
    Build a small deterministic sparse embedding for local development.

    This is a hashed lexical embedding, not a production semantic model. It gives
    the project a replaceable vector-search interface before external embedding
    providers or model downloads are introduced.
    """
    vector: Dict[str, float] = {}
    for feature, weight in _features(tokenize(text)).items():
        bucket, sign = _hash_feature(feature, dimensions)
        vector[bucket] = vector.get(bucket, 0.0) + weight * sign

    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}

    return {
        bucket: round(value / norm, 6)
        for bucket, value in vector.items()
        if value != 0
    }


def _embed_text_remote(text: str) -> List[float]:
    if not settings.embedding_api_key:
        raise EmbeddingClientError("Embedding API key is not configured.")

    base_url = settings.embedding_base_url.rstrip("/")
    request_url = f"{base_url}/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": text,
    }
    body = json.dumps(payload).encode("utf-8")

    request = Request(
        request_url,
        data=body,
        headers={
            "Authorization": f"Bearer {settings.embedding_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.embedding_timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingClientError(f"Embedding API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise EmbeddingClientError(f"Failed to reach Embedding API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise EmbeddingClientError("Embedding API returned invalid JSON.") from exc

    try:
        embedding = response_data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingClientError("Embedding API response did not include an embedding.") from exc

    if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
        raise EmbeddingClientError("Embedding API returned an invalid embedding vector.")

    return [float(value) for value in embedding]


def _load_local_model():
    global _LOCAL_MODEL

    if _LOCAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingClientError(
                "Local embedding provider requires sentence-transformers."
            ) from exc

        try:
            _LOCAL_MODEL = SentenceTransformer(
                settings.local_embedding_model,
                local_files_only=settings.local_embedding_local_files_only,
            )
        except Exception as exc:
            raise EmbeddingClientError(
                f"Failed to load local embedding model: {settings.local_embedding_model}"
            ) from exc

    return _LOCAL_MODEL


def _embed_text_local_model(text: str) -> List[float]:
    model = _load_local_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return [float(value) for value in embedding.tolist()]


def embed_text(text: str) -> Embedding:
    """
    Return an embedding using the configured provider.

    Providers:
    - local_model: sentence-transformers model running on this machine.
    - openai_compatible: OpenAI-compatible /embeddings API.
    - local_hashed: deterministic sparse embedding fallback.
    """
    provider = settings.embedding_provider.lower()

    if provider in {"local_model", "sentence_transformers"}:
        try:
            return _embed_text_local_model(text)
        except EmbeddingClientError:
            if not settings.embedding_fallback_to_local:
                raise
            return _embed_text_local(text)

    if provider == "openai_compatible":
        try:
            return _embed_text_remote(text)
        except EmbeddingClientError:
            if not settings.embedding_fallback_to_local:
                raise
            return _embed_text_local(text)

    if provider == "local_hashed":
        return _embed_text_local(text)

    if not settings.embedding_fallback_to_local:
        raise EmbeddingClientError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return _embed_text_local(text)


def current_embedding_model() -> str:
    provider = settings.embedding_provider.lower()
    if provider in {"local_model", "sentence_transformers"}:
        return settings.local_embedding_model
    if provider == "openai_compatible" and is_embedding_configured():
        return settings.embedding_model
    return LOCAL_EMBEDDING_MODEL


def embedding_model_for_vector(vector: Embedding) -> str:
    provider = settings.embedding_provider.lower()
    if isinstance(vector, list):
        if provider in {"local_model", "sentence_transformers"}:
            return settings.local_embedding_model
        return settings.embedding_model
    return LOCAL_EMBEDDING_MODEL


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    if not left or not right:
        return 0.0

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return 0.0

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0

    if len(left) > len(right):
        left, right = right, left

    return sum(value * right.get(bucket, 0.0) for bucket, value in left.items())
