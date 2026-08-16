from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)
    retrieval_mode: Literal["keyword", "vector", "hybrid", "rerank"] = "keyword"
    min_score: float | None = Field(default=None, ge=0.0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "How many annual leave days are available?",
                    "top_k": 3,
                    "retrieval_mode": "hybrid",
                    "min_score": 0.2,
                }
            ]
        }
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized


class Citation(BaseModel):
    chunk_id: str
    document_id: str | None = None
    filename: str | None = None
    position: int | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    score: float | None = None
    retrieval_mode: str | None = None
    context_role: Literal["retrieved", "adjacent"] = "retrieved"
    expanded_from_chunk_id: str | None = None


class RetrievedContext(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    position: int
    chunk_index: int | None = None
    page_number: int | None = None
    content: str
    token_count: int | None = None
    created_at: str | None = None
    score: float | None = None
    retrieval_mode: str | None = None
    context_role: Literal["retrieved", "adjacent"] = "retrieved"
    expanded_from_chunk_id: str | None = None
    keyword_score: float | None = None
    vector_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    pre_rerank_score: float | None = None
    question_coverage: float | None = None
    token_density: float | None = None
    number_match: float | None = None
    embedding_model: str | None = None
    query_embedding_model: str | None = None
    vector_store_backend: str | None = None


class ChatData(BaseModel):
    request_id: str
    question: str
    retrieval_mode: Literal["keyword", "vector", "hybrid", "rerank"]
    min_score: float
    answer: str
    answer_mode: Literal["llm", "local_fallback", "no_context"]
    model: str | None = None
    llm_error: str | None = None
    citations: list[Citation]
    contexts: list[RetrievedContext]


class ChatResponse(BaseModel):
    success: bool
    data: ChatData
    message: str
