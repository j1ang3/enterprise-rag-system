import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


DATASET_SCHEMA_VERSION = "business_policy_eval.v1"
QUERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_FIELDS = {
    "query_id",
    "question",
    "expected_answer",
    "expected_sources",
    "expected_source_match",
    "expected_citation_chunk_ids",
    "expected_keywords",
    "category",
    "difficulty",
    "should_answer",
}


@dataclass(frozen=True)
class EvaluationCase:
    query_id: str
    question: str
    expected_answer: str
    expected_sources: Tuple[str, ...]
    expected_source_match: str
    expected_citation_chunk_ids: Tuple[str, ...]
    expected_keywords: Tuple[str, ...]
    category: str
    difficulty: str
    answerable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_sources": list(self.expected_sources),
            "expected_source_match": self.expected_source_match,
            "expected_citation_chunk_ids": list(self.expected_citation_chunk_ids),
            "expected_keywords": list(self.expected_keywords),
            "category": self.category,
            "difficulty": self.difficulty,
            "should_answer": self.answerable,
        }

    def expected_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.expected_answer,
            "documents": list(self.expected_sources),
            "document_match": self.expected_source_match,
            "chunk_ids": list(self.expected_citation_chunk_ids),
            "required_keywords": list(self.expected_keywords),
        }


@dataclass(frozen=True)
class EvaluationDataset:
    path: Path
    sha256: str
    cases: Tuple[EvaluationCase, ...]

    def identity(self, *, project_root: Path | None = None) -> Dict[str, Any]:
        recorded_path = self.path
        if project_root is not None:
            try:
                recorded_path = self.path.resolve().relative_to(project_root.resolve())
            except ValueError:
                recorded_path = self.path.resolve()
        return {
            "path": str(recorded_path),
            "sha256": self.sha256,
            "schema_version": DATASET_SCHEMA_VERSION,
            "query_count": len(self.cases),
            "query_ids": [case.query_id for case in self.cases],
            "ordering": "JSONL non-empty line order",
            "unknown_fields": "rejected",
            "answerable_count": sum(case.answerable for case in self.cases),
            "unanswerable_count": sum(not case.answerable for case in self.cases),
        }


def _non_empty_string(value: object, *, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} {field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, field: str, location: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{location} {field} must be a list of strings")
    normalized = tuple(
        _non_empty_string(item, field=field, location=location)
        for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{location} {field} must not contain duplicates")
    return normalized


def _normalize_case(raw: object, *, position: int, line_number: int) -> EvaluationCase:
    location = f"case {position} (line {line_number})"
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be a JSON object")

    unknown = sorted(set(raw) - ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"{location} contains unknown fields: {', '.join(unknown)}")

    query_id = raw.get("query_id", f"q{position:03d}")
    query_id = _non_empty_string(query_id, field="query_id", location=location)
    if not QUERY_ID_PATTERN.fullmatch(query_id):
        raise ValueError(f"{location} query_id contains unsupported characters")

    answerable = raw.get("should_answer")
    if not isinstance(answerable, bool):
        raise ValueError(f"{location} should_answer must be a boolean")

    expected_sources = _string_tuple(
        raw.get("expected_sources"),
        field="expected_sources",
        location=location,
    )
    if answerable and not expected_sources:
        raise ValueError(f"{location} answerable case requires expected_sources")
    if not answerable and expected_sources:
        raise ValueError(f"{location} unanswerable case must not define expected_sources")

    source_match = raw.get("expected_source_match", "any")
    if source_match not in {"any", "all"}:
        raise ValueError(f"{location} expected_source_match must be 'any' or 'all'")

    return EvaluationCase(
        query_id=query_id,
        question=_non_empty_string(
            raw.get("question"), field="question", location=location
        ),
        expected_answer=_non_empty_string(
            raw.get("expected_answer"), field="expected_answer", location=location
        ),
        expected_sources=expected_sources,
        expected_source_match=source_match,
        expected_citation_chunk_ids=_string_tuple(
            raw.get("expected_citation_chunk_ids", []),
            field="expected_citation_chunk_ids",
            location=location,
        ),
        expected_keywords=_string_tuple(
            raw.get("expected_keywords"),
            field="expected_keywords",
            location=location,
        ),
        category=_non_empty_string(
            raw.get("category"), field="category", location=location
        ),
        difficulty=_non_empty_string(
            raw.get("difficulty"), field="difficulty", location=location
        ),
        answerable=answerable,
    )


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"evaluation dataset not found: {resolved}")

    cases = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at {resolved}:{line_number}"
            ) from exc
        cases.append(
            _normalize_case(
                raw,
                position=len(cases) + 1,
                line_number=line_number,
            )
        )

    if not cases:
        raise ValueError("evaluation dataset must not be empty")
    query_ids = [case.query_id for case in cases]
    if len(set(query_ids)) != len(query_ids):
        duplicates = sorted(
            query_id for query_id in set(query_ids) if query_ids.count(query_id) > 1
        )
        raise ValueError(f"duplicate query_id values: {', '.join(duplicates)}")

    return EvaluationDataset(
        path=resolved,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        cases=tuple(cases),
    )
