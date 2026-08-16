from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "annual leave policy", "top_k": 5, "min_score": 0.2}
            ]
        }
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class SemanticSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    position: int
    chunk_index: int | None = None
    page_number: int | None = None
    content: str
    token_count: int | None = None
    score: float
    retrieval_mode: str
    embedding_model: str | None = None
    query_embedding_model: str | None = None
    vector_store_backend: str | None = None
    created_at: str | None = None


class SemanticSearchData(BaseModel):
    query: str
    top_k: int
    min_score: float
    result_count: int
    results: list[SemanticSearchResult]


class SemanticSearchResponse(BaseModel):
    success: bool
    data: SemanticSearchData
    message: str


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"query": "HR-2026 leave policy", "top_k": 5}]
        }
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class HybridSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    position: int | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    content: str
    text: str
    token_count: int | None = None
    metadata: dict[str, Any]
    source: str
    score: float | None = None
    retrieval_mode: str | None = None
    bm25_score: float | None = None
    embedding_model: str | None = None
    query_embedding_model: str | None = None
    vector_store_backend: str | None = None
    created_at: str | None = None


class HybridSearchData(BaseModel):
    query: str
    top_k_per_source: int
    result_count: int
    results: list[HybridSearchResult]


class HybridSearchResponse(BaseModel):
    success: bool
    data: HybridSearchData
    message: str


class RankFusionSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_depth: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "HR-2026 leave policy",
                    "top_k": 5,
                    "candidate_depth": 20,
                    "rrf_k": 60,
                }
            ]
        }
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class RankFusionSearchResult(HybridSearchResult):
    fused_score: float
    matched_sources: list[str]
    source_ranks: dict[str, int]
    source_scores: dict[str, float | None]


class RankFusionSearchData(BaseModel):
    query: str
    top_k: int
    candidate_depth: int
    rrf_k: int
    result_count: int
    results: list[RankFusionSearchResult]


class RankFusionSearchResponse(BaseModel):
    success: bool
    data: RankFusionSearchData
    message: str
