from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Enterprise RAG Knowledge Base"
    app_version: str = "0.1.0"
    llm_provider: str = "openai_compatible"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "OLLAMA_API_KEY",
        ),
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices(
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "OLLAMA_BASE_URL",
        ),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "LLM_MODEL",
            "OPENAI_MODEL",
            "OLLAMA_MODEL",
        ),
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1)
    llm_seed: int | None = Field(default=None, ge=0)
    llm_reasoning_effort: str = ""
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    embedding_provider: str = "local_model"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_local_files_only: bool = True
    embedding_timeout_seconds: int = 30
    embedding_fallback_to_local: bool = True
    # Week 4 uses FAISS as the primary local vector index. The backend adapter
    # retains numpy/JSON fallbacks when a FAISS index cannot be loaded.
    vector_store_backend: str = "faiss"
    vector_store_external_provider: str = ""
    vector_store_url: str = ""
    vector_store_collection: str = "enterprise_rag_chunks"
    vector_store_api_key: str = ""
    vector_store_timeout_seconds: int = 10
    keyword_min_score: float = 0.2
    vector_min_score: float = 0.2
    hybrid_min_score: float = 0.2
    hybrid_keyword_weight: float = 0.8
    hybrid_vector_weight: float = 0.2
    rerank_min_score: float = 0.2
    rerank_candidate_multiplier: int = 6
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_local_files_only: bool = True
    context_window_chunks: int = 1
    max_expanded_contexts: int = 8
    rag_structured_logging_enabled: bool = True
    rag_structured_log_path: Path = Path("logs/rag.jsonl")
    # Layered is the secure application default. Historical W8/W9 evaluators pass
    # baseline explicitly so their frozen prompt identity remains reproducible.
    rag_security_mode: Literal["baseline", "layered"] = "layered"
    # W10 application metadata lives in PostgreSQL. Empty defaults keep imports
    # side-effect free; database operations fail explicitly when not configured.
    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    test_database_url: str = Field(default="", validation_alias="TEST_DATABASE_URL")
    # Authentication fails closed at the /auth boundary when this remains empty.
    # SecretStr reduces the chance of accidental disclosure through repr/logging.
    jwt_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="JWT_SECRET_KEY",
    )
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        validation_alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
TEXT_DIR = STORAGE_DIR / "texts"
INDEX_DIR = STORAGE_DIR / "index"
KNOWLEDGE_BASE_FILE = INDEX_DIR / "chunks.json"
VECTOR_INDEX_FILE = INDEX_DIR / "vectors.json"

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
