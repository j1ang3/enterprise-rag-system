from pathlib import Path
from typing import AbstractSet, List

from app.retrieval.base import RetrievalResult
from app.services.vector_store import search_vector_chunks


class VectorRetriever:
    """Adapt the existing vector-search function to the Retriever interface."""

    source = "vector"

    def __init__(
        self,
        *,
        index_path: Path | None = None,
        min_score: float = 0.0,
        allowed_document_ids: AbstractSet[str],
    ) -> None:
        self.index_path = index_path
        self.min_score = min_score
        self.allowed_document_ids = frozenset(allowed_document_ids)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip():
            return []

        return search_vector_chunks(
            query,
            top_k=top_k,
            index_path=self.index_path,
            min_score=self.min_score,
            allowed_document_ids=self.allowed_document_ids,
        )
