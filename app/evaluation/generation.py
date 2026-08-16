import re
from typing import Any, Dict, Iterable, Mapping, Sequence


METRIC_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "required_keyword_match": {
        "version": 1,
        "range": [0.0, 1.0],
        "deterministic": True,
        "llm_judge": False,
        "definition": (
            "Case-insensitive substring match for every human-authored required keyword. "
            "This is a lexical answer-correctness proxy, not semantic correctness."
        ),
    },
    "document_citation": {
        "version": 1,
        "range": [0.0, 1.0],
        "deterministic": True,
        "llm_judge": False,
        "definition": (
            "Set precision, recall, F1 and exact match between cited filenames and "
            "human-authored expected document labels."
        ),
    },
    "strict_chunk_citation_recall": {
        "version": 1,
        "range": [0.0, 1.0],
        "deterministic": True,
        "llm_judge": False,
        "definition": (
            "Recall of expected citation chunk IDs, only for cases with explicit "
            "human-authored chunk labels."
        ),
    },
    "groundedness": {
        "status": "not_automated",
        "reason": "No claim-level labels or validated judge setup exists in W8-T1.",
    },
    "answer_relevance": {
        "status": "not_automated",
        "reason": "No validated semantic relevance evaluator or judge setup exists in W8-T1.",
    },
}


def normalize_lexical_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def calculate_required_keyword_proxy(
    answer: str,
    expected_keywords: Sequence[str],
) -> Dict[str, Any]:
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    if not expected_keywords:
        return {
            "status": "not_available_no_required_keywords",
            "matched": None,
            "recall": None,
            "matched_keywords": [],
            "missing_keywords": [],
        }

    normalized_answer = normalize_lexical_text(answer)
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if normalize_lexical_text(keyword) in normalized_answer
    ]
    missing_keywords = [
        keyword for keyword in expected_keywords if keyword not in matched_keywords
    ]
    return {
        "status": "evaluated",
        "matched": not missing_keywords,
        "recall": len(matched_keywords) / len(expected_keywords),
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
    }


def calculate_set_metrics(
    actual_ids: Iterable[str],
    expected_ids: Iterable[str],
) -> Dict[str, Any]:
    actual_set = {identity for identity in actual_ids if identity}
    expected_set = {identity for identity in expected_ids if identity}
    if not expected_set:
        raise ValueError("expected_ids must contain at least one identity")

    matched = actual_set & expected_set
    precision = len(matched) / len(actual_set) if actual_set else 0.0
    recall = len(matched) / len(expected_set)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": actual_set == expected_set,
        "matched_ids": sorted(matched),
        "missing_ids": sorted(expected_set - actual_set),
        "unexpected_ids": sorted(actual_set - expected_set),
    }


def calculate_citation_metrics(
    citations: Sequence[Mapping[str, Any]],
    *,
    expected_documents: Sequence[str],
    expected_chunk_ids: Sequence[str],
) -> Dict[str, Any]:
    cited_documents = [
        citation.get("filename")
        for citation in citations
        if isinstance(citation.get("filename"), str)
    ]
    cited_chunk_ids = [
        citation.get("chunk_id")
        for citation in citations
        if isinstance(citation.get("chunk_id"), str)
    ]
    result: Dict[str, Any] = {
        "ground_truth_level": "document_filename",
        "document": calculate_set_metrics(cited_documents, expected_documents),
        "strict_chunk": {
            "status": "not_available_no_chunk_labels",
            "recall": None,
        },
    }
    if expected_chunk_ids:
        chunk_metrics = calculate_set_metrics(cited_chunk_ids, expected_chunk_ids)
        result["strict_chunk"] = {
            "status": "evaluated",
            "recall": chunk_metrics["recall"],
            "matched_ids": chunk_metrics["matched_ids"],
            "missing_ids": chunk_metrics["missing_ids"],
        }
    return result
