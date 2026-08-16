import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Sequence
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval import (
    aggregate_retrieval_results,
    calculate_retrieval_metrics,
)
from app.retrieval import DEFAULT_RRF_K, HybridRetriever, VectorRetriever
from app.services.access_control import get_readable_document_ids
from app.services.knowledge_base import build_bm25_retriever
from scripts.evaluate_rag import bootstrap_documents, load_dataset


DEFAULT_DATASET = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
DEFAULT_BOOTSTRAP_DOCS = (
    PROJECT_ROOT / "eval_docs" / "hr_policy.md",
    PROJECT_ROOT / "eval_docs" / "expense_policy.md",
    PROJECT_ROOT / "eval_docs" / "security_policy.md",
    PROJECT_ROOT / "eval_docs" / "product_faq.md",
)
DEFAULT_CHUNK_INDEX = PROJECT_ROOT / "storage" / "eval" / "week06_retrieval_chunks.json"
DEFAULT_VECTOR_INDEX = PROJECT_ROOT / "storage" / "eval" / "week06_retrieval_vectors.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "results" / "W6-T4-retrieval-evaluation.json"
DEFAULT_K_VALUES = (1, 3, 5)
DEFAULT_CANDIDATE_DEPTH = 5


@dataclass(frozen=True)
class RetrievalMethod:
    name: str
    retrieve: Callable[[str, int], List[Dict[str, Any]]]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_retrieval_methods(
    chunk_index_path: Path,
    vector_index_path: Path,
    *,
    candidate_depth: int,
    rrf_k: int,
    allowed_document_ids: frozenset[str],
) -> Dict[str, RetrievalMethod]:
    """Construct all compared methods through their public Retriever interfaces."""
    vector = VectorRetriever(
        index_path=vector_index_path,
        min_score=0.0,
        allowed_document_ids=allowed_document_ids,
    )
    bm25 = build_bm25_retriever(
        chunk_index_path,
        allowed_document_ids=allowed_document_ids,
    )
    hybrid = HybridRetriever(
        [vector, bm25],
        allowed_document_ids=allowed_document_ids,
    )

    return {
        "vector": RetrievalMethod("vector", vector.retrieve),
        "bm25": RetrievalMethod("bm25", bm25.retrieve),
        "hybrid_rrf": RetrievalMethod(
            "hybrid_rrf",
            lambda query, top_k: hybrid.retrieve_fused(
                query,
                top_k,
                candidate_depth=candidate_depth,
                rrf_k=rrf_k,
            ),
        ),
    }


def validate_examples(examples: Sequence[Mapping[str, Any]]) -> None:
    if not examples:
        raise ValueError("evaluation dataset must not be empty")

    for position, example in enumerate(examples, start=1):
        question = example.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"query {position} must contain a non-empty question")

        expected_sources = example.get("expected_sources")
        if not isinstance(expected_sources, list) or any(
            not isinstance(source, str) or not source.strip()
            for source in expected_sources
        ):
            raise ValueError(
                f"query {position} expected_sources must be a list of non-empty strings"
            )
        if bool(example.get("should_answer", True)) and not expected_sources:
            raise ValueError(
                f"query {position} is answerable but has no document relevance labels"
            )


def _validate_results(results: Sequence[Mapping[str, Any]], method: str) -> None:
    for rank, result in enumerate(results, start=1):
        for field in ("chunk_id", "filename"):
            value = result.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{method} result at rank {rank} is missing non-empty {field}"
                )


def evaluate_method(
    method: RetrievalMethod,
    examples: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    max_k = max(k_values)
    results = []

    for position, example in enumerate(examples, start=1):
        query = str(example["question"])
        started = perf_counter()
        retrieved = method.retrieve(query, max_k)
        latency_ms = (perf_counter() - started) * 1000.0
        _validate_results(retrieved, method.name)

        relevant_documents = list(example["expected_sources"])
        metrics_status = (
            "evaluated"
            if relevant_documents
            else "not_applicable_no_relevant_documents"
        )
        retrieved_documents = [str(result["filename"]) for result in retrieved]
        metrics_by_k = (
            {
                str(k): {
                    "k": k,
                    **calculate_retrieval_metrics(
                        retrieved_documents,
                        relevant_documents,
                        top_k=k,
                    ),
                }
                for k in k_values
            }
            if relevant_documents
            else {}
        )

        relevant_chunk_ids = example.get("expected_citation_chunk_ids")
        results.append(
            {
                "query_id": f"q{position:03d}",
                "query": query,
                "method": method.name,
                "category": example.get("category", "uncategorized"),
                "difficulty": example.get("difficulty", "unknown"),
                "should_answer": bool(example.get("should_answer", True)),
                "retrieved_chunk_ids": [result["chunk_id"] for result in retrieved],
                "retrieved_document_labels": retrieved_documents,
                "retrieval_scores": [result.get("score") for result in retrieved],
                "relevant_document_labels": relevant_documents,
                "relevant_chunk_ids": (
                    list(relevant_chunk_ids)
                    if isinstance(relevant_chunk_ids, list) and relevant_chunk_ids
                    else None
                ),
                "relevance_label_type": "document_filename",
                "metrics_status": metrics_status,
                "metrics_by_k": metrics_by_k,
                "latency_ms": latency_ms,
            }
        )
    return results


def build_comparisons(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    comparison_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Record measured query-level advantages and misses without tuning rankings."""
    by_method = {
        method: {result["query_id"]: result for result in results}
        for method, results in all_results.items()
    }
    query_ids = list(by_method["vector"])
    comparisons = {
        "bm25_better_than_vector": [],
        "vector_better_than_bm25": [],
        "hybrid_better_than_vector": [],
        "hybrid_worse_than_vector": [],
        "all_methods_missed": [],
    }
    metric_key = str(comparison_k)

    def quality(result: Mapping[str, Any]) -> tuple[float, float]:
        metrics = result["metrics_by_k"][metric_key]
        return float(metrics["recall"]), float(metrics["reciprocal_rank"])

    for query_id in query_ids:
        vector = by_method["vector"][query_id]
        if vector["metrics_status"] != "evaluated":
            continue
        bm25 = by_method["bm25"][query_id]
        hybrid = by_method["hybrid_rrf"][query_id]
        vector_quality = quality(vector)
        bm25_quality = quality(bm25)
        hybrid_quality = quality(hybrid)

        detail = {
            "query_id": query_id,
            "query": vector["query"],
            "category": vector["category"],
            "vector": {
                "quality": vector_quality,
                "retrieved_chunk_ids": vector["retrieved_chunk_ids"],
            },
            "bm25": {
                "quality": bm25_quality,
                "retrieved_chunk_ids": bm25["retrieved_chunk_ids"],
            },
            "hybrid_rrf": {
                "quality": hybrid_quality,
                "retrieved_chunk_ids": hybrid["retrieved_chunk_ids"],
            },
        }
        if bm25_quality > vector_quality:
            comparisons["bm25_better_than_vector"].append(detail)
        if vector_quality > bm25_quality:
            comparisons["vector_better_than_bm25"].append(detail)
        if hybrid_quality > vector_quality:
            comparisons["hybrid_better_than_vector"].append(detail)
        if hybrid_quality < vector_quality:
            comparisons["hybrid_worse_than_vector"].append(detail)
        if vector_quality == bm25_quality == hybrid_quality == (0.0, 0.0):
            comparisons["all_methods_missed"].append(detail)

    return comparisons


def build_strict_chunk_subset(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """Summarize the small subset that has explicit human chunk labels."""
    subset_results: Dict[str, List[Dict[str, Any]]] = {}
    for method, results in all_results.items():
        method_subset = []
        for result in results:
            relevant_chunk_ids = result.get("relevant_chunk_ids")
            if not relevant_chunk_ids:
                continue
            method_subset.append(
                {
                    **result,
                    "metrics_status": "evaluated",
                    "metrics_by_k": {
                        str(k): {
                            "k": k,
                            **calculate_retrieval_metrics(
                                result["retrieved_chunk_ids"],
                                relevant_chunk_ids,
                                top_k=k,
                            ),
                        }
                        for k in k_values
                    },
                }
            )
        subset_results[method] = method_subset

    query_ids = [result["query_id"] for result in subset_results["vector"]]
    return {
        "label_type": "chunk_id",
        "query_count": len(query_ids),
        "query_ids": query_ids,
        "limitation": "Diagnostic only: the fixed dataset has strict chunk labels for three queries.",
        "summary": {
            method: aggregate_retrieval_results(results, k_values=k_values)
            for method, results in subset_results.items()
        },
    }


def run_evaluation(
    *,
    dataset_path: Path,
    bootstrap_docs: Sequence[Path],
    chunk_index_path: Path,
    vector_index_path: Path,
    output_path: Path,
    k_values: Sequence[int],
    candidate_depth: int,
    rrf_k: int,
    user_id: UUID,
) -> Dict[str, Any]:
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if candidate_depth < max(k_values):
        raise ValueError("candidate_depth must be at least the largest evaluation K")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    examples = load_dataset(dataset_path)
    validate_examples(examples)
    indexed_documents = bootstrap_documents(
        bootstrap_docs,
        chunk_index_path,
        vector_index_path,
    )
    allowed_document_ids = get_readable_document_ids(user_id)
    methods = build_retrieval_methods(
        chunk_index_path,
        vector_index_path,
        candidate_depth=candidate_depth,
        rrf_k=rrf_k,
        allowed_document_ids=allowed_document_ids,
    )
    results_by_method = {
        name: evaluate_method(method, examples, k_values=k_values)
        for name, method in methods.items()
    }
    summaries = {
        name: aggregate_retrieval_results(results, k_values=k_values)
        for name, results in results_by_method.items()
    }

    payload = {
        "task": "W6-T4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth": {
            "source": str(dataset_path.relative_to(PROJECT_ROOT)),
            "label_type": "document_filename",
            "metric_scope": "answerable queries with one or more expected_sources",
            "strict_chunk_label_coverage": sum(
                1 for example in examples if example.get("expected_citation_chunk_ids")
            ),
            "dataset_sha256": _file_sha256(dataset_path),
        },
        "configuration": {
            "authorization_user_id": str(user_id),
            "query_count": len(examples),
            "methods": list(methods),
            "k_values": list(k_values),
            "hybrid_candidate_depth": candidate_depth,
            "rrf_k": rrf_k,
            "vector_min_score": 0.0,
            "bootstrap_documents": [
                str(path.relative_to(PROJECT_ROOT)) for path in bootstrap_docs
            ],
            "bootstrap_document_sha256": {
                str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
                for path in bootstrap_docs
            },
            "indexed_documents": indexed_documents,
            "indexed_chunk_count": sum(
                int(document["chunk_count"]) for document in indexed_documents
            ),
        },
        "summary": summaries,
        "comparisons_by_k": {
            str(k): build_comparisons(results_by_method, comparison_k=k)
            for k in k_values
        },
        "strict_chunk_labeled_subset": build_strict_chunk_subset(
            results_by_method,
            k_values=k_values,
        ),
        "results": results_by_method,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Vector, BM25, and RRF Hybrid retrieval on fixed labels."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--bootstrap-docs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_BOOTSTRAP_DOCS),
    )
    parser.add_argument("--chunk-index-path", type=Path, default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--vector-index-path", type=Path, default=DEFAULT_VECTOR_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=DEFAULT_CANDIDATE_DEPTH,
    )
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL user whose readable documents define the evaluation corpus.",
    )
    args = parser.parse_args()

    payload = run_evaluation(
        dataset_path=_resolve(args.dataset),
        bootstrap_docs=[_resolve(path) for path in args.bootstrap_docs],
        chunk_index_path=_resolve(args.chunk_index_path),
        vector_index_path=_resolve(args.vector_index_path),
        output_path=_resolve(args.output),
        k_values=tuple(sorted(set(args.k_values))),
        candidate_depth=args.candidate_depth,
        rrf_k=args.rrf_k,
        user_id=args.user_id,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote detailed results to {_resolve(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
