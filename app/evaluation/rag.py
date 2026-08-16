import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Sequence
from uuid import UUID

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.generation import (
    METRIC_DEFINITIONS as GENERATION_METRIC_DEFINITIONS,
    calculate_citation_metrics,
    calculate_required_keyword_proxy,
)
from app.evaluation.retrieval import calculate_retrieval_metrics
from app.services.rag_service import answer_question
from app.services.search_service import RerankedHybridConfig


FORMAL_LLM_PROVIDER = "ollama"
FORMAL_LLM_MODEL = "qwen3:8b"
ARTIFACT_VERSION = 1
RAGCallable = Callable[..., Dict[str, Any]]


RETRIEVAL_METRIC_DEFINITIONS = {
    "ground_truth_level": "document_filename_primary_with_optional_strict_chunk_labels",
    "hit_rate_at_k": "At least one expected document appears in the first K final retrieved chunks.",
    "recall_at_k": "Unique expected documents found in the first K divided by all expected documents.",
    "mrr_at_k": "Reciprocal rank of the first expected document within the first K.",
    "precision_at_k": "Not reported: document-only labels do not judge every retrieved chunk.",
    "ndcg_at_k": "Not reported: the dataset has no graded relevance labels.",
}


@dataclass(frozen=True)
class RAGEvaluationConfig:
    formal: bool
    retrieval_mode: str
    top_k: int
    metric_k_values: tuple[int, ...]
    index_path: Path
    vector_index_path: Path
    reranked_hybrid: RerankedHybridConfig
    llm_metadata: Mapping[str, Any]
    user_id: UUID
    security_mode: str = "baseline"

    def __post_init__(self) -> None:
        if self.retrieval_mode != "hybrid_rerank":
            raise ValueError("W8-T1 requires retrieval_mode='hybrid_rerank'")
        if self.top_k != self.reranked_hybrid.final_top_k:
            raise ValueError("top_k must equal the frozen reranker final_top_k")
        if not self.metric_k_values:
            raise ValueError("metric_k_values must not be empty")
        if tuple(sorted(set(self.metric_k_values))) != self.metric_k_values:
            raise ValueError("metric_k_values must be sorted and unique")
        if any(k <= 0 or k > self.top_k for k in self.metric_k_values):
            raise ValueError("metric K values must be between 1 and final top_k")
        if self.formal:
            provider = str(self.llm_metadata.get("provider", "")).casefold()
            model = self.llm_metadata.get("model")
            if provider != FORMAL_LLM_PROVIDER or model != FORMAL_LLM_MODEL:
                raise ValueError(
                    "formal evaluation requires Ollama with qwen3:8b"
                )
            identity = self.llm_metadata.get("model_identity")
            if not isinstance(identity, Mapping) or not identity.get("digest"):
                raise ValueError("formal evaluation requires resolved model identity")
        if self.security_mode not in {"baseline", "layered"}:
            raise ValueError("security_mode must be baseline or layered")

    def resolved_dict(self, *, project_root: Path) -> Dict[str, Any]:
        return {
            "formal": self.formal,
            "retrieval_mode": self.retrieval_mode,
            "final_top_k": self.top_k,
            "metric_k_values": list(self.metric_k_values),
            "chunk_index_path": _relative_path(self.index_path, project_root),
            "vector_index_path": _relative_path(self.vector_index_path, project_root),
            "reranked_hybrid": self.reranked_hybrid.to_dict(),
            "security_mode": self.security_mode,
            "llm": dict(self.llm_metadata),
            "authorization_user_id": str(self.user_id),
        }


class RAGEvaluationRunner:
    def __init__(
        self,
        configuration: RAGEvaluationConfig,
        *,
        rag_callable: RAGCallable = answer_question,
    ) -> None:
        self.configuration = configuration
        self._rag_callable = rag_callable

    def run_case(self, case: EvaluationCase) -> Dict[str, Any]:
        started = perf_counter()
        try:
            rag_result = self._rag_callable(
                case.question,
                self.configuration.top_k,
                retrieval_mode=self.configuration.retrieval_mode,
                min_score=None,
                index_path=self.configuration.index_path,
                vector_index_path=self.configuration.vector_index_path,
                reranked_hybrid_config=self.configuration.reranked_hybrid,
                security_mode=self.configuration.security_mode,
                user_id=self.configuration.user_id,
            )
        except Exception as exc:
            return _failure_result(
                case,
                runtime_ms=(perf_counter() - started) * 1000.0,
                failure_stage="rag",
                error_category="rag_error",
                error=exc,
            )

        try:
            actual = _normalize_rag_result(rag_result)
            failure = self._formal_output_failure(actual)
            if failure is not None:
                return _failure_result(
                    case,
                    runtime_ms=(perf_counter() - started) * 1000.0,
                    failure_stage="generation",
                    error_category="provider_error",
                    error=RuntimeError(failure),
                    actual=actual,
                )
            metrics = _evaluate_successful_case(
                case,
                actual,
                k_values=self.configuration.metric_k_values,
            )
        except Exception as exc:
            return _failure_result(
                case,
                runtime_ms=(perf_counter() - started) * 1000.0,
                failure_stage="evaluation",
                error_category="evaluation_error",
                error=exc,
                actual=_best_effort_actual(rag_result),
            )

        return {
            "query_id": case.query_id,
            "question": case.question,
            "answerable": case.answerable,
            "category": case.category,
            "difficulty": case.difficulty,
            "status": "success",
            "expected": case.expected_dict(),
            "actual": actual,
            "metrics": metrics,
            "evaluation_runtime_ms": (perf_counter() - started) * 1000.0,
            "error": None,
        }

    def _formal_output_failure(self, actual: Mapping[str, Any]) -> str | None:
        if not self.configuration.formal:
            return None
        answer_mode = actual["answer_mode"]
        contexts = actual["contexts"]
        if contexts and answer_mode != "llm":
            return (
                "formal qwen3:8b evaluation received a non-LLM fallback: "
                f"{actual.get('llm_error') or answer_mode}"
            )
        if answer_mode == "llm" and actual.get("model") != FORMAL_LLM_MODEL:
            return "formal evaluation output model differed from qwen3:8b"
        return None

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        run_id: str,
        run_metadata: Mapping[str, Any],
        project_root: Path,
    ) -> Dict[str, Any]:
        results = [self.run_case(case) for case in dataset.cases]
        return {
            "artifact_version": ARTIFACT_VERSION,
            "task": "W8-T1",
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "formal": self.configuration.formal,
            "status": "completed",
            "run_metadata": dict(run_metadata),
            "dataset_identity": dataset.identity(project_root=project_root),
            "resolved_configuration": self.configuration.resolved_dict(
                project_root=project_root
            ),
            "metric_definitions": {
                "retrieval": RETRIEVAL_METRIC_DEFINITIONS,
                "generation": GENERATION_METRIC_DEFINITIONS,
                "failure_denominator": (
                    "Execution failures are reported separately and excluded from "
                    "quality metric denominators."
                ),
                "unanswerable_boundary": (
                    "Raw outputs are preserved, but ordinary answer correctness and "
                    "citation quality aggregates exclude unanswerable cases."
                ),
            },
            "aggregate": aggregate_results(results, self.configuration.metric_k_values),
            "results": results,
        }


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_mapping_list(value: object, *, field: str) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"RAG result {field} must be a list")
    normalized = []
    for position, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"RAG result {field}[{position}] must be an object")
        normalized.append(dict(item))
    return normalized


def _normalize_rag_result(rag_result: object) -> Dict[str, Any]:
    if not isinstance(rag_result, Mapping):
        raise ValueError("RAG service must return an object")
    answer = rag_result.get("answer")
    answer_mode = rag_result.get("answer_mode")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("RAG result answer must be non-empty")
    if answer_mode not in {"llm", "local_fallback", "no_context"}:
        raise ValueError("RAG result answer_mode is invalid")

    contexts = _normalize_mapping_list(rag_result.get("contexts"), field="contexts")
    citations = _normalize_mapping_list(
        rag_result.get("citations"), field="citations"
    )
    evidence = rag_result.get("retrieval_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("hybrid_rerank RAG result must include retrieval_evidence")
    pre_rerank = _normalize_mapping_list(
        evidence.get("candidates_before_rerank"),
        field="retrieval_evidence.candidates_before_rerank",
    )
    final_retrieved = _normalize_mapping_list(
        evidence.get("results_after_rerank"),
        field="retrieval_evidence.results_after_rerank",
    )

    return {
        "request_id": rag_result.get("request_id"),
        "answer": answer.strip(),
        "answer_mode": answer_mode,
        "model": rag_result.get("model"),
        "llm_error": rag_result.get("llm_error"),
        "citations": citations,
        "retrieval": {
            "candidates_before_rerank": pre_rerank,
            "final_chunks": final_retrieved,
            "configuration": dict(evidence.get("configuration", {})),
        },
        "contexts": contexts,
        "latency_ms": {
            "retrieval": rag_result.get("retrieval_latency_ms"),
            "rerank": rag_result.get("rerank_latency_ms"),
            "context_build": rag_result.get("context_build_latency_ms"),
            "generation": rag_result.get("generation_latency_ms"),
            "llm": rag_result.get("llm_latency_ms"),
            "total": rag_result.get("total_latency_ms"),
        },
        "llm_usage": rag_result.get("llm_usage"),
        "security": rag_result.get("security"),
        "runtime_event": rag_result.get("runtime_event"),
    }


def _best_effort_actual(rag_result: object) -> Dict[str, Any] | None:
    if not isinstance(rag_result, Mapping):
        return None
    return {
        "request_id": rag_result.get("request_id"),
        "answer": rag_result.get("answer"),
        "answer_mode": rag_result.get("answer_mode"),
        "model": rag_result.get("model"),
        "llm_error": rag_result.get("llm_error"),
        "citations": rag_result.get("citations"),
        "contexts": rag_result.get("contexts"),
        "retrieval_evidence": rag_result.get("retrieval_evidence"),
        "llm_usage": rag_result.get("llm_usage"),
        "security": rag_result.get("security"),
        "runtime_event": rag_result.get("runtime_event"),
    }


def _evaluate_successful_case(
    case: EvaluationCase,
    actual: Mapping[str, Any],
    *,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    if not case.answerable:
        return {
            "retrieval": {"status": "not_applicable_no_relevant_documents"},
            "generation": {
                "status": "deferred_to_W8_T2",
                "answer_correctness": None,
                "citation": None,
            },
        }

    final_chunks = actual["retrieval"]["final_chunks"]
    retrieved_documents = [
        chunk["filename"]
        for chunk in final_chunks
        if isinstance(chunk.get("filename"), str)
    ]
    retrieved_chunk_ids = [
        chunk["chunk_id"]
        for chunk in final_chunks
        if isinstance(chunk.get("chunk_id"), str)
    ]
    document_metrics = {
        str(k): calculate_retrieval_metrics(
            retrieved_documents,
            case.expected_sources,
            top_k=k,
        )
        for k in k_values
    }
    strict_chunk_metrics = {
        "status": "not_available_no_chunk_labels",
        "metrics_by_k": {},
    }
    if case.expected_citation_chunk_ids:
        strict_chunk_metrics = {
            "status": "evaluated",
            "metrics_by_k": {
                str(k): calculate_retrieval_metrics(
                    retrieved_chunk_ids,
                    case.expected_citation_chunk_ids,
                    top_k=k,
                )
                for k in k_values
            },
        }

    return {
        "retrieval": {
            "status": "evaluated",
            "ground_truth_level": "document_filename",
            "metrics_by_k": document_metrics,
            "strict_chunk": strict_chunk_metrics,
        },
        "generation": {
            "status": "evaluated_deterministic_proxies",
            "required_keyword_proxy": calculate_required_keyword_proxy(
                actual["answer"], case.expected_keywords
            ),
            "citation": calculate_citation_metrics(
                actual["citations"],
                expected_documents=case.expected_sources,
                expected_chunk_ids=case.expected_citation_chunk_ids,
            ),
            "groundedness": {"status": "not_automated"},
            "answer_relevance": {"status": "not_automated"},
        },
    }


def _failure_result(
    case: EvaluationCase,
    *,
    runtime_ms: float,
    failure_stage: str,
    error_category: str,
    error: Exception,
    actual: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "query_id": case.query_id,
        "question": case.question,
        "answerable": case.answerable,
        "category": case.category,
        "difficulty": case.difficulty,
        "status": "failed",
        "expected": case.expected_dict(),
        "actual": dict(actual) if actual is not None else None,
        "metrics": None,
        "evaluation_runtime_ms": runtime_ms,
        "error": {
            "failure_stage": failure_stage,
            "category": error_category,
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _average(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    successful = [result for result in results if result["status"] == "success"]
    failed = [result for result in results if result["status"] == "failed"]
    answerable = [result for result in successful if result["answerable"]]
    unanswerable = [result for result in successful if not result["answerable"]]

    retrieval_by_k: Dict[str, Dict[str, Any]] = {}
    strict_retrieval_by_k: Dict[str, Dict[str, Any]] = {}
    for k in k_values:
        key = str(k)
        document_rows = [result["metrics"]["retrieval"]["metrics_by_k"][key] for result in answerable]
        retrieval_by_k[key] = {
            "sample_count": len(document_rows),
            "hit_rate": _average([1.0 if row["hit"] else 0.0 for row in document_rows]),
            "recall": _average([float(row["recall"]) for row in document_rows]),
            "mrr": _average([float(row["reciprocal_rank"]) for row in document_rows]),
        }
        strict_rows = [
            result["metrics"]["retrieval"]["strict_chunk"]["metrics_by_k"][key]
            for result in answerable
            if result["metrics"]["retrieval"]["strict_chunk"]["status"] == "evaluated"
        ]
        strict_retrieval_by_k[key] = {
            "sample_count": len(strict_rows),
            "hit_rate": _average([1.0 if row["hit"] else 0.0 for row in strict_rows]),
            "recall": _average([float(row["recall"]) for row in strict_rows]),
            "mrr": _average([float(row["reciprocal_rank"]) for row in strict_rows]),
        }

    keyword_rows = [
        result["metrics"]["generation"]["required_keyword_proxy"]
        for result in answerable
        if result["metrics"]["generation"]["required_keyword_proxy"]["status"] == "evaluated"
    ]
    citation_rows = [
        result["metrics"]["generation"]["citation"]["document"]
        for result in answerable
    ]
    strict_citation_rows = [
        result["metrics"]["generation"]["citation"]["strict_chunk"]
        for result in answerable
        if result["metrics"]["generation"]["citation"]["strict_chunk"]["status"] == "evaluated"
    ]

    answer_mode_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}
    for result in results:
        actual = result.get("actual")
        if not isinstance(actual, Mapping):
            continue
        mode = actual.get("answer_mode")
        if mode:
            answer_mode_counts[str(mode)] = answer_mode_counts.get(str(mode), 0) + 1
        model = actual.get("model")
        if model:
            model_counts[str(model)] = model_counts.get(str(model), 0) + 1

    return {
        "case_counts": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "answerable_total": sum(bool(result["answerable"]) for result in results),
            "answerable_successful": len(answerable),
            "unanswerable_total": sum(not bool(result["answerable"]) for result in results),
            "unanswerable_successful": len(unanswerable),
        },
        "quality_denominators": {
            "retrieval_and_generation": "successful answerable cases",
            "failed_execution": "excluded and reported separately",
            "unanswerable": "excluded from ordinary generation quality; raw output preserved",
        },
        "retrieval": {
            "ground_truth_level": "document_filename",
            "metrics_by_k": retrieval_by_k,
            "strict_chunk_labeled_subset": strict_retrieval_by_k,
        },
        "generation": {
            "required_keyword_proxy": {
                "sample_count": len(keyword_rows),
                "match_rate": _average([1.0 if row["matched"] else 0.0 for row in keyword_rows]),
                "mean_required_keyword_recall": _average([float(row["recall"]) for row in keyword_rows]),
            },
            "document_citation": {
                "sample_count": len(citation_rows),
                "exact_match_rate": _average([1.0 if row["exact_match"] else 0.0 for row in citation_rows]),
                "mean_precision": _average([float(row["precision"]) for row in citation_rows]),
                "mean_recall": _average([float(row["recall"]) for row in citation_rows]),
                "mean_f1": _average([float(row["f1"]) for row in citation_rows]),
            },
            "strict_chunk_citation_recall": {
                "sample_count": len(strict_citation_rows),
                "mean_recall": _average([float(row["recall"]) for row in strict_citation_rows]),
            },
            "groundedness": "not_automated",
            "answer_relevance": "not_automated",
        },
        "execution": {
            "answer_mode_counts": answer_mode_counts,
            "model_counts": model_counts,
            "failed_query_ids": [result["query_id"] for result in failed],
            "failures": [result["error"] for result in failed],
            "mean_evaluation_runtime_ms": _average(
                [float(result["evaluation_runtime_ms"]) for result in results]
            ),
        },
        "unanswerable": {
            "raw_output_count": len(unanswerable),
            "formal_behavior_scoring": "deferred_to_W8_T2",
        },
    }


def write_artifact(
    artifact: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"evaluation artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_evaluation_report(
    artifact: Mapping[str, Any],
    *,
    artifact_path: str,
) -> str:
    aggregate = artifact["aggregate"]
    counts = aggregate["case_counts"]
    config = artifact["resolved_configuration"]
    llm = config["llm"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Run ID: `{artifact['run_id']}`",
        f"- Formal: `{str(artifact['formal']).lower()}`",
        f"- Dataset: `{artifact['dataset_identity']['path']}`",
        f"- Dataset SHA-256: `{artifact['dataset_identity']['sha256']}`",
        f"- Cases: {counts['total']} total / {counts['successful']} successful / {counts['failed']} failed",
        f"- Answerable / unanswerable: {counts['answerable_total']} / {counts['unanswerable_total']}",
        f"- LLM: `{llm['provider']}` / `{llm['model']}`",
        f"- Model digest: `{llm.get('model_identity', {}).get('digest')}`",
        f"- Retrieval: `{config['retrieval_mode']}`; final top_k={config['final_top_k']}",
        f"- Artifact: `{artifact_path}`",
        "",
        "## Aggregate Retrieval Results",
        "",
        "| K | Samples | Hit Rate | Recall | MRR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key, metrics in aggregate["retrieval"]["metrics_by_k"].items():
        lines.append(
            f"| {key} | {metrics['sample_count']} | {_fmt(metrics['hit_rate'])} | "
            f"{_fmt(metrics['recall'])} | {_fmt(metrics['mrr'])} |"
        )

    generation = aggregate["generation"]
    keyword = generation["required_keyword_proxy"]
    citation = generation["document_citation"]
    strict_citation = generation["strict_chunk_citation_recall"]
    lines.extend(
        [
            "",
            "## Aggregate Generation Results",
            "",
            f"- Required-keyword correctness proxy: {_fmt(keyword['match_rate'])} "
            f"({keyword['sample_count']} successful answerable cases).",
            f"- Mean required-keyword recall: {_fmt(keyword['mean_required_keyword_recall'])}.",
            f"- Document citation exact match: {_fmt(citation['exact_match_rate'])}.",
            f"- Document citation precision / recall / F1: {_fmt(citation['mean_precision'])} / "
            f"{_fmt(citation['mean_recall'])} / {_fmt(citation['mean_f1'])}.",
            f"- Strict chunk citation recall: {_fmt(strict_citation['mean_recall'])} "
            f"({strict_citation['sample_count']} labeled cases).",
            "- Groundedness: not formally automated in W8-T1.",
            "- Answer relevance: not formally automated in W8-T1.",
            "- No LLM judge or RAGAS was used.",
            "",
            "## Failed Cases",
            "",
        ]
    )
    failed_ids = aggregate["execution"]["failed_query_ids"]
    lines.append(
        "None." if not failed_ids else ", ".join(f"`{query_id}`" for query_id in failed_ids)
    )
    lines.extend(
        [
            "",
            "## Boundaries and Limitations",
            "",
            "- Primary retrieval and citation Ground Truth is document-level; only three cases have strict chunk labels.",
            "- The required-keyword metric is a deterministic lexical proxy, not semantic correctness.",
            "- Execution failures are excluded from quality denominators and reported separately.",
            "- Unanswerable raw outputs are preserved, but hallucination/refusal scoring is deferred to W8-T2.",
            "- Structured production logging is deferred to W8-T3; failure root-cause classification is deferred to W8-T4.",
            "- Reproducibility means conditions and identities are recorded; local model output is not promised bit-for-bit identical.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
