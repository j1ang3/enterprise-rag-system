import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from typing import Any, Protocol

from app.core.config import settings
from app.retrieval.base import RetrievalResult


QueryCandidatePair = tuple[str, str]
CrossEncoderLoader = Callable[..., Any]


class PairScorer(Protocol):
    """Score query/candidate text pairs in one joint inference step."""

    def score_pairs(self, pairs: Sequence[QueryCandidatePair]) -> Sequence[float]:
        ...


class RerankerModelError(RuntimeError):
    """Raised when the reranking model cannot load or return valid scores."""


def _load_cross_encoder(model_name: str, *, local_files_only: bool) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, local_files_only=local_files_only)


class CrossEncoderScorer:
    """Lazy sentence-transformers CrossEncoder adapter reused across calls."""

    def __init__(
        self,
        model_name: str,
        *,
        local_files_only: bool = True,
        model_loader: CrossEncoderLoader = _load_cross_encoder,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model_loader = model_loader
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                self._model = self._model_loader(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
            except Exception as exc:
                raise RerankerModelError(
                    f"Failed to load reranking model: {self.model_name}"
                ) from exc
        return self._model

    def score_pairs(self, pairs: Sequence[QueryCandidatePair]) -> Sequence[float]:
        model = self._get_model()
        try:
            return model.predict(list(pairs))
        except Exception as exc:
            raise RerankerModelError("Reranking model scoring failed") from exc


class SemanticReranker:
    """Reorder retrieved candidates using joint query/chunk relevance scores."""

    def __init__(self, scorer: PairScorer) -> None:
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        normalized_query = self._validate_query(query)
        self._validate_top_k(top_k)

        candidate_copies, pairs = self._prepare_candidates(
            normalized_query,
            candidates,
        )
        if not candidate_copies:
            return []

        scores = self._score_pairs(pairs)
        reranked = [
            {**candidate, "rerank_score": score}
            for candidate, score in zip(candidate_copies, scores, strict=True)
        ]

        # Python's sort is stable, so equal model scores retain the input/RRF order.
        reranked.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
        return reranked if top_k is None else reranked[:top_k]

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        return query.strip()

    @staticmethod
    def _validate_top_k(top_k: int | None) -> None:
        if top_k is None:
            return
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer or None")

    @staticmethod
    def _prepare_candidates(
        query: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[list[RetrievalResult], list[QueryCandidatePair]]:
        candidate_copies: list[RetrievalResult] = []
        pairs: list[QueryCandidatePair] = []
        seen_chunk_ids: set[str] = set()

        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"candidate at index {index} must be a mapping")

            chunk_id = candidate.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"candidate at index {index} must contain a non-empty chunk_id"
                )
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"duplicate candidate chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            candidate_text = candidate.get("content")
            if not isinstance(candidate_text, str):
                candidate_text = candidate.get("text")
            if not isinstance(candidate_text, str) or not candidate_text.strip():
                raise ValueError(
                    f"candidate {chunk_id} must contain non-empty content or text"
                )

            candidate_copies.append(deepcopy(dict(candidate)))
            pairs.append((query, candidate_text))

        return candidate_copies, pairs

    def _score_pairs(self, pairs: Sequence[QueryCandidatePair]) -> list[float]:
        try:
            raw_scores = list(self._scorer.score_pairs(pairs))
        except RerankerModelError:
            raise
        except Exception as exc:
            raise RerankerModelError("Reranking model scoring failed") from exc

        if len(raw_scores) != len(pairs):
            raise RerankerModelError(
                "Reranking model returned a different number of scores than candidates"
            )

        scores: list[float] = []
        for raw_score in raw_scores:
            if isinstance(raw_score, bool):
                raise RerankerModelError("Reranking model returned a non-numeric score")
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise RerankerModelError(
                    "Reranking model returned a non-numeric score"
                ) from exc
            if not math.isfinite(score):
                raise RerankerModelError("Reranking model returned a non-finite score")
            scores.append(score)
        return scores


@lru_cache(maxsize=1)
def get_default_reranker() -> SemanticReranker:
    """Return one process-level reranker whose lazy model is reused."""
    return SemanticReranker(
        CrossEncoderScorer(
            settings.reranker_model_name,
            local_files_only=settings.reranker_local_files_only,
        )
    )
