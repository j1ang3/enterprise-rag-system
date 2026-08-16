from typing import Any, Dict, List, Protocol


RetrievalResult = Dict[str, Any]


class Retriever(Protocol):
    """Small interface shared by retrieval components."""

    source: str

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        ...
