import re
from collections import Counter
from typing import Any, Dict, List

from app.services.embeddings import tokenize


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return set(NUMBER_PATTERN.findall(text))


def _coverage_score(question_tokens: List[str], content_tokens: List[str]) -> float:
    if not question_tokens or not content_tokens:
        return 0.0

    question_counts = Counter(question_tokens)
    content_counts = Counter(content_tokens)
    matched = sum(min(count, content_counts.get(token, 0)) for token, count in question_counts.items())
    return matched / sum(question_counts.values())


def _density_score(question_tokens: List[str], content_tokens: List[str]) -> float:
    if not question_tokens or not content_tokens:
        return 0.0

    question_set = set(question_tokens)
    matched = sum(1 for token in content_tokens if token in question_set)
    return matched / len(content_tokens)


def _number_score(question: str, content: str) -> float:
    question_numbers = _numbers(question)
    if not question_numbers:
        return 0.0

    content_numbers = _numbers(content)
    if not content_numbers:
        return 0.0

    return len(question_numbers & content_numbers) / len(question_numbers)


def rerank_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Lightweight local reranker.

    This is intentionally simple and explainable. It is not a replacement for a
    trained cross-encoder reranker, but it gives the project a reranking stage
    that can be evaluated and replaced later.
    """
    question_tokens = tokenize(question)
    reranked = []

    for chunk in chunks:
        content = chunk.get("content", "")
        content_tokens = tokenize(content)
        base_score = float(chunk.get("score", 0.0))
        coverage = _coverage_score(question_tokens, content_tokens)
        density = _density_score(question_tokens, content_tokens)
        number_match = _number_score(question, content)
        rerank_score = (
            0.55 * base_score
            + 0.30 * coverage
            + 0.10 * density
            + 0.05 * number_match
        )

        if rerank_score > 0:
            reranked.append(
                {
                    **chunk,
                    "score": round(rerank_score, 4),
                    "rerank_score": round(rerank_score, 4),
                    "pre_rerank_score": round(base_score, 4),
                    "question_coverage": round(coverage, 4),
                    "token_density": round(density, 4),
                    "number_match": round(number_match, 4),
                    "retrieval_mode": "rerank",
                }
            )

    reranked.sort(key=lambda chunk: chunk["score"], reverse=True)
    return _select_reranked_context_candidates(reranked, top_k, min_score)


def _select_reranked_context_candidates(
    reranked: List[Dict[str, Any]],
    top_k: int,
    min_score: float,
) -> List[Dict[str, Any]]:
    selected = [chunk for chunk in reranked if chunk["score"] > min_score][:top_k]
    if len(selected) >= top_k or not reranked:
        return selected

    selected_chunk_ids = {chunk["chunk_id"] for chunk in selected}
    selected_document_ids = {chunk["document_id"] for chunk in selected}
    relaxed_min_score = min_score * 0.75

    for chunk in reranked:
        if len(selected) >= top_k:
            break
        if chunk["chunk_id"] in selected_chunk_ids:
            continue
        if chunk["document_id"] in selected_document_ids:
            continue
        if chunk["score"] <= relaxed_min_score:
            continue

        selected.append(chunk)
        selected_chunk_ids.add(chunk["chunk_id"])
        selected_document_ids.add(chunk["document_id"])

    for chunk in reranked:
        if len(selected) >= top_k:
            break
        if chunk["chunk_id"] in selected_chunk_ids:
            continue
        if chunk["score"] <= relaxed_min_score:
            continue

        selected.append(chunk)
        selected_chunk_ids.add(chunk["chunk_id"])
        selected_document_ids.add(chunk["document_id"])

    return selected
