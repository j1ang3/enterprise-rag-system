import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.rag import (
    RAGEvaluationConfig,
    RAGEvaluationRunner,
    render_evaluation_report,
    write_artifact,
)
from app.evaluation.unanswerable import (
    UnanswerableEvaluationRunner,
    load_unanswerable_cases,
    load_unanswerable_manifest,
    render_unanswerable_report,
    validate_w8_t1_source_artifact,
    verify_absence_against_corpus,
)
from app.services.knowledge_base import index_document
from app.services.llm_client import get_llm_runtime_metadata, is_llm_configured
from app.services.prompts import SYSTEM_PROMPT
from app.services.rag_service import answer_question
from app.services.search_service import RerankedHybridConfig
from app.services.storage_paths import get_document_storage_paths
from app.services.text_loader import extract_text


DEFAULT_DATASET = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
DEFAULT_W7_MANIFEST = PROJECT_ROOT / "evals" / "retrieval_evaluation_config.json"
DEFAULT_FORMAL_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "evaluation_runs"
DEFAULT_EVALUATION_REPORT = PROJECT_ROOT / "evals/results/generated_reports/evaluation-report.md"
DEFAULT_UNANSWERABLE_MANIFEST = (
    PROJECT_ROOT / "evals" / "unanswerable_evaluation_config.json"
)
DEFAULT_UNANSWERABLE_OUTPUT_DIR = (
    PROJECT_ROOT / "evals" / "results" / "unanswerable_runs"
)
DEFAULT_UNANSWERABLE_REPORT = (
    PROJECT_ROOT / "evals/results/generated_reports/unanswerable-evaluation-report.md"
)


def disable_llm_for_evaluation() -> None:
    """Prevent .env credentials from turning an offline baseline into live traffic."""
    settings.llm_api_key = ""


def default_eval_index(retrieval_mode: str, dataset: Path | str = DEFAULT_DATASET) -> Path:
    dataset_stem = Path(dataset).stem
    return PROJECT_ROOT / "storage" / "eval" / f"{retrieval_mode}_{dataset_stem}_chunks.json"


def default_eval_vector_index(retrieval_mode: str, dataset: Path | str = DEFAULT_DATASET) -> Path:
    dataset_stem = Path(dataset).stem
    return PROJECT_ROOT / "storage" / "eval" / f"{retrieval_mode}_{dataset_stem}_vectors.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Backward-compatible list view over the validated W8 dataset loader."""
    return [case.to_dict() for case in load_evaluation_dataset(path).cases]


def bootstrap_documents(
    paths: Iterable[Path],
    index_path: Path,
    vector_index_path: Path,
) -> List[Dict[str, Any]]:
    indexed = []
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("[]", encoding="utf-8")
    vector_index_path.write_text("[]", encoding="utf-8")

    for path in paths:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Bootstrap document not found: {resolved}")

        text = extract_text(resolved)
        document_id = f"eval-{_slug(resolved.stem)}"
        chunks = index_document(
            document_id,
            resolved.name,
            text,
            index_path=index_path,
            vector_index_path=vector_index_path,
        )
        indexed.append(
            {
                "document_id": document_id,
                "filename": resolved.name,
                "chunk_count": len(chunks),
            }
        )
    return indexed


def _expected_source_hit(
    contexts: List[Dict[str, Any]],
    expected_sources: List[str],
    expected_source_match: str = "any",
) -> bool:
    if not expected_sources:
        return not contexts
    retrieved_filenames = {context.get("filename") for context in contexts}
    if expected_source_match == "all":
        return all(source in retrieved_filenames for source in expected_sources)
    return any(source in retrieved_filenames for source in expected_sources)


def _expected_citation_source_hit(
    citations: List[Dict[str, Any]],
    expected_sources: List[str],
    expected_source_match: str = "any",
) -> bool:
    if not expected_sources:
        return not citations
    citation_filenames = {citation.get("filename") for citation in citations}
    if expected_source_match == "all":
        return all(source in citation_filenames for source in expected_sources)
    return any(source in citation_filenames for source in expected_sources)


def _expected_citation_chunk_hit(
    citations: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
) -> bool:
    if not expected_chunk_ids:
        return True
    citation_chunk_ids = {citation.get("chunk_id") for citation in citations}
    return all(chunk_id in citation_chunk_ids for chunk_id in expected_chunk_ids)


def _keyword_hit(answer: str, expected_keywords: List[str]) -> bool:
    if not expected_keywords:
        return True

    normalized_answer = answer.lower()
    return all(keyword.lower() in normalized_answer for keyword in expected_keywords)


def _citation_hit(citations: List[Dict[str, Any]], should_answer: bool) -> bool:
    if should_answer:
        return bool(citations)
    return not citations


def evaluate_example(
    example: Dict[str, Any],
    top_k: int,
    index_path: Optional[Path],
    vector_index_path: Optional[Path],
    retrieval_mode: str,
    min_score: Optional[float],
    require_llm: bool = False,
    *,
    user_id: UUID,
) -> Dict[str, Any]:
    question = example["question"]
    should_answer = bool(example.get("should_answer", True))
    expected_sources = example.get("expected_sources", [])
    expected_source_match = example.get("expected_source_match", "any")
    expected_citation_sources = example.get("expected_citation_sources", expected_sources)
    expected_citation_source_match = example.get(
        "expected_citation_source_match",
        expected_source_match,
    )
    expected_citation_chunk_ids = example.get("expected_citation_chunk_ids", [])
    expected_keywords = example.get("expected_keywords", [])

    rag_result = answer_question(
        question,
        top_k,
        retrieval_mode=retrieval_mode,
        min_score=min_score,
        index_path=index_path,
        vector_index_path=vector_index_path or get_document_storage_paths().vectors_file,
        user_id=user_id,
    )
    contexts = rag_result["contexts"]
    answer = rag_result["answer"]
    citations = rag_result.get("citations", [])

    if require_llm and contexts and rag_result["answer_mode"] != "llm":
        error = rag_result.get("llm_error") or "unknown LLM failure"
        raise RuntimeError(
            f"Formal LLM evaluation stopped for question {question!r}: {error}"
        )

    source_hit = _expected_source_hit(contexts, expected_sources, expected_source_match)
    keyword_hit = _keyword_hit(answer, expected_keywords)
    no_answer_hit = bool(contexts) is False if not should_answer else True
    citation_hit = _citation_hit(citations, should_answer)
    citation_source_hit = _expected_citation_source_hit(
        citations,
        expected_citation_sources,
        expected_citation_source_match,
    )
    citation_chunk_hit = _expected_citation_chunk_hit(citations, expected_citation_chunk_ids)

    return {
        "question": question,
        "category": example.get("category", "uncategorized"),
        "difficulty": example.get("difficulty", "unknown"),
        "should_answer": should_answer,
        "answer_mode": rag_result["answer_mode"],
        "model": rag_result.get("model"),
        "llm_error": rag_result.get("llm_error"),
        "retrieval_mode": retrieval_mode,
        "min_score": rag_result["min_score"],
        "retrieved_sources": [context.get("filename") for context in contexts],
        "retrieved_chunk_ids": [context.get("chunk_id") for context in contexts],
        "retrieval_scores": [context.get("score") for context in contexts],
        "citation_sources": [citation.get("filename") for citation in citations],
        "citation_chunk_ids": [citation.get("chunk_id") for citation in citations],
        "retrieval_latency_ms": rag_result["retrieval_latency_ms"],
        "generation_latency_ms": rag_result["generation_latency_ms"],
        "total_latency_ms": rag_result["total_latency_ms"],
        "source_hit": source_hit,
        "keyword_hit": keyword_hit,
        "no_answer_hit": no_answer_hit,
        "citation_hit": citation_hit,
        "citation_source_hit": citation_source_hit,
        "citation_chunk_hit": citation_chunk_hit,
        "passed": (
            source_hit
            and keyword_hit
            and no_answer_hit
            and citation_hit
            and citation_source_hit
            and citation_chunk_hit
        ),
        "answer": answer,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    answerable = [result for result in results if result["should_answer"]]
    unanswerable = [result for result in results if not result["should_answer"]]

    def rate(items: List[Dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return round(sum(1 for item in items if item[key]) / len(items), 4)

    def average(items: List[Dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return round(sum(float(item[key]) for item in items) / len(items), 3)

    answer_mode_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}
    for result in results:
        mode = result["answer_mode"]
        answer_mode_counts[mode] = answer_mode_counts.get(mode, 0) + 1
        model = result.get("model")
        if model:
            model_counts[model] = model_counts.get(model, 0) + 1

    return {
        "total_examples": total,
        "passed": sum(1 for result in results if result["passed"]),
        "pass_rate": rate(results, "passed"),
        "retrieval_recall_at_k": rate(answerable, "source_hit"),
        "answer_keyword_accuracy": rate(answerable, "keyword_hit"),
        "answer_correctness_proxy": rate(answerable, "keyword_hit"),
        "no_answer_accuracy": rate(unanswerable, "no_answer_hit"),
        "citation_presence_accuracy": rate(results, "citation_hit"),
        "citation_source_accuracy": rate(results, "citation_source_hit"),
        "citation_chunk_accuracy": rate(results, "citation_chunk_hit"),
        "average_retrieval_latency_ms": average(results, "retrieval_latency_ms"),
        "average_generation_latency_ms": average(results, "generation_latency_ms"),
        "average_total_latency_ms": average(results, "total_latency_ms"),
        "answer_mode_counts": answer_mode_counts,
        "model_counts": model_counts,
    }


def _repository_state() -> Dict[str, Any]:
    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = git("status", "--short")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_entry_count": len(status.splitlines()) if status else 0,
    }


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"w8-t1-{timestamp}-qwen3-8b"


def _new_unanswerable_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"w8-t2-{timestamp}-qwen3-8b"


def _resolve_formal_inputs(
    args: argparse.Namespace,
    *,
    llm_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    # Import lazily because the W7 runner imports this module's compatibility
    # dataset/bootstrap functions.
    from scripts.evaluate_reranked_retrieval import validate_frozen_manifest

    if args.disable_llm:
        raise ValueError("formal evaluation cannot disable the LLM")
    if args.retrieval_mode != "hybrid_rerank":
        raise ValueError("formal W8 evaluation requires --retrieval-mode hybrid_rerank")

    manifest_path = args.w7_manifest.resolve()
    validated = validate_frozen_manifest(manifest_path)
    manifest = validated["manifest"]
    frozen = validated["config"]
    dataset_path = validated["dataset_path"]
    chunk_index_path = validated["chunk_index_path"]
    vector_index_path = validated["vector_index_path"]

    if args.dataset.resolve() != dataset_path:
        raise ValueError("formal dataset must match the frozen W7 dataset identity")
    if args.index_path is not None and args.index_path.resolve() != chunk_index_path:
        raise ValueError("formal chunk index must match the frozen W7 index")
    if (
        args.vector_index_path is not None
        and args.vector_index_path.resolve() != vector_index_path
    ):
        raise ValueError("formal vector index must match the frozen W7 index")
    if args.bootstrap_docs:
        raise ValueError("formal evaluation uses the frozen corpus and cannot bootstrap docs")

    reranked_configuration = RerankedHybridConfig(
        per_source_candidate_depth=frozen["per_source_candidate_depth"],
        rrf_k=frozen["rrf_k"],
        rerank_candidate_count=frozen["reranker_candidate_count"],
        final_top_k=frozen["operational_final_top_k"],
    )
    if args.top_k != reranked_configuration.final_top_k:
        raise ValueError(
            f"formal top_k must equal frozen value {reranked_configuration.final_top_k}"
        )

    dataset = load_evaluation_dataset(dataset_path)
    configuration = RAGEvaluationConfig(
        formal=True,
        retrieval_mode="hybrid_rerank",
        top_k=reranked_configuration.final_top_k,
        metric_k_values=tuple(
            k
            for k in frozen["metric_k_values"]
            if k <= reranked_configuration.final_top_k
        ),
        index_path=chunk_index_path,
        vector_index_path=vector_index_path,
        reranked_hybrid=reranked_configuration,
        llm_metadata=llm_metadata,
        user_id=args.user_id,
    )
    corpus_identity = {
        "chunk_size": manifest["corpus"]["chunk_size"],
        "chunk_overlap": manifest["corpus"]["chunk_overlap"],
        "indexed_chunk_count": manifest["corpus"]["indexed_chunk_count"],
        "documents": manifest["corpus"]["documents"],
        "chunk_index": manifest["corpus"]["chunk_index"],
        "vector_index": manifest["corpus"]["vector_index"],
        "faiss_index": manifest["corpus"]["faiss_index"],
        "faiss_metadata": manifest["corpus"]["faiss_metadata"],
    }
    return {
        "manifest_path": manifest_path,
        "validated": validated,
        "manifest": manifest,
        "dataset": dataset,
        "configuration": configuration,
        "corpus_identity": corpus_identity,
    }


def run_formal_evaluation(
    args: argparse.Namespace,
    *,
    llm_metadata: Mapping[str, Any],
) -> int:
    resolved = _resolve_formal_inputs(args, llm_metadata=llm_metadata)
    manifest_path = resolved["manifest_path"]
    validated = resolved["validated"]
    manifest = resolved["manifest"]
    dataset = resolved["dataset"]
    configuration = resolved["configuration"]

    run_id = args.run_id or _new_run_id()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / f"{run_id}.json"
    report_path = args.report_path.resolve()
    if artifact_path.exists() and not args.overwrite:
        raise FileExistsError(f"evaluation artifact already exists: {artifact_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"evaluation report already exists: {report_path}; use --overwrite"
        )
    run_metadata = {
        "command": [sys.executable, *sys.argv],
        "repository": _repository_state(),
        "w7_frozen_manifest": {
            "path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": validated["manifest_sha256"],
            "experiment_id": manifest["experiment_id"],
        },
        "corpus_identity": resolved["corpus_identity"],
        "prompt_identity": {
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "context_format": "app.services.prompts.format_contexts",
        },
        "randomness": {
            "configured_temperature": llm_metadata.get("temperature"),
            "seed": None,
            "guarantee": "conditions are reproducible; output is not promised bit-for-bit identical",
        },
    }
    artifact = RAGEvaluationRunner(configuration).run(
        dataset,
        run_id=run_id,
        run_metadata=run_metadata,
        project_root=PROJECT_ROOT,
    )
    write_artifact(artifact, artifact_path, overwrite=args.overwrite)
    report = render_evaluation_report(
        artifact,
        artifact_path=str(artifact_path.relative_to(PROJECT_ROOT)),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\nFormal W8-T1 evaluation summary")
    print(json.dumps(artifact["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nWrote artifact to {artifact_path}")
    print(f"Wrote report to {report_path}")
    return 1 if artifact["aggregate"]["case_counts"]["failed"] else 0


def run_formal_unanswerable_evaluation(
    args: argparse.Namespace,
    *,
    llm_metadata: Mapping[str, Any],
) -> int:
    resolved = _resolve_formal_inputs(args, llm_metadata=llm_metadata)
    dataset = resolved["dataset"]
    configuration = resolved["configuration"]
    manifest_bundle = load_unanswerable_manifest(
        args.unanswerable_manifest,
        project_root=PROJECT_ROOT,
    )
    unanswerable_manifest = manifest_bundle["manifest"]
    if manifest_bundle["stable_dataset_path"] != dataset.path:
        raise ValueError("W8-T2 stable dataset differs from frozen W8-T1/W7 input")
    if manifest_bundle["w7_manifest_path"] != resolved["manifest_path"]:
        raise ValueError("W8-T2 W7 manifest identity differs from the formal input")
    if manifest_bundle["chunk_index_path"] != configuration.index_path:
        raise ValueError("W8-T2 chunk snapshot differs from the formal input")

    source_w8_t1 = validate_w8_t1_source_artifact(
        manifest_bundle["source_w8_t1_artifact_path"],
        expected_dataset_sha256=dataset.sha256,
    )
    cases = load_unanswerable_cases(
        manifest_bundle["case_file_path"],
        stable_dataset=dataset,
    )
    expected_case_ids = unanswerable_manifest["case_file"].get("query_ids")
    if [case.query_id for case in cases] != expected_case_ids:
        raise ValueError("W8-T2 case order/identity differs from the frozen manifest")
    stable_case_ids = [
        case.query_id for case in cases if case.source == "stable_dataset"
    ]
    if stable_case_ids != unanswerable_manifest["stable_dataset"].get(
        "stable_unanswerable_query_ids"
    ):
        raise ValueError("W8-T2 stable unanswerable subset drifted")

    dataset_by_id = {case.query_id: case for case in dataset.cases}
    control_ids = unanswerable_manifest["answerable_controls"].get("query_ids")
    if not isinstance(control_ids, list) or not control_ids:
        raise ValueError("W8-T2 answerable controls are missing")
    try:
        controls = [dataset_by_id[query_id] for query_id in control_ids]
    except KeyError as exc:
        raise ValueError(f"W8-T2 control query is missing: {exc.args[0]}") from exc
    if any(not case.answerable for case in controls):
        raise ValueError("W8-T2 controls must all be answerable")

    absence_verification = verify_absence_against_corpus(
        cases,
        document_paths=manifest_bundle["document_paths"],
        chunk_index_path=manifest_bundle["chunk_index_path"],
    )

    run_id = args.run_id or _new_unanswerable_run_id()
    output_dir = args.unanswerable_output_dir.resolve()
    artifact_path = output_dir / f"{run_id}.json"
    report_path = args.unanswerable_report_path.resolve()
    if artifact_path.exists() and not args.overwrite:
        raise FileExistsError(f"unanswerable artifact already exists: {artifact_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"unanswerable report already exists: {report_path}; use --overwrite"
        )

    validated = resolved["validated"]
    manifest = resolved["manifest"]
    run_metadata = {
        "command": [sys.executable, *sys.argv],
        "repository": _repository_state(),
        "prompt_identity": {
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "context_format": "app.services.prompts.format_contexts",
        },
        "randomness": {
            "configured_temperature": llm_metadata.get("temperature"),
            "seed": None,
            "guarantee": "conditions are reproducible; output is not promised bit-for-bit identical",
        },
        "classification": {
            "rubric_version": unanswerable_manifest["rubric"]["version"],
            "llm_judge": False,
            "manual_review_performed": False,
        },
    }
    source_identities = {
        "unanswerable_manifest": {
            "path": str(manifest_bundle["manifest_path"].relative_to(PROJECT_ROOT)),
            "sha256": manifest_bundle["manifest_sha256"],
            "experiment_id": unanswerable_manifest["experiment_id"],
        },
        "source_w8_t1_artifact": {
            **unanswerable_manifest["source_w8_t1_artifact"],
            "run_id": source_w8_t1["run_id"],
        },
        "stable_dataset": dataset.identity(project_root=PROJECT_ROOT),
        "unanswerable_case_file": unanswerable_manifest["case_file"],
        "answerable_control_ids": control_ids,
        "w7_frozen_manifest": {
            "path": str(resolved["manifest_path"].relative_to(PROJECT_ROOT)),
            "sha256": validated["manifest_sha256"],
            "experiment_id": manifest["experiment_id"],
        },
        "corpus": resolved["corpus_identity"],
    }
    artifact = UnanswerableEvaluationRunner(
        RAGEvaluationRunner(configuration),
        rubric=unanswerable_manifest["rubric"],
    ).run(
        unanswerable_cases=cases,
        absence_verification=absence_verification,
        control_cases=controls,
        run_id=run_id,
        run_metadata=run_metadata,
        source_identities=source_identities,
        project_root=PROJECT_ROOT,
    )
    write_artifact(artifact, artifact_path, overwrite=args.overwrite)
    report = render_unanswerable_report(
        artifact,
        artifact_path=str(artifact_path.relative_to(PROJECT_ROOT)),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\nFormal W8-T2 unanswerable evaluation summary")
    print(json.dumps(artifact["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nWrote artifact to {artifact_path}")
    print(f"Wrote report to {report_path}")
    failed = (
        artifact["aggregate"]["unanswerable"]["failed"]
        + artifact["aggregate"]["answerable_controls"]["failed"]
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the seed RAG evaluation dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL User whose ownership/ACL permissions apply to evaluation.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=["keyword", "vector", "hybrid", "rerank", "hybrid_rerank"],
        default="keyword",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        help="Optional chunk index to evaluate. When --bootstrap-docs is used, defaults to a retrieval-mode-specific file under storage/eval.",
    )
    parser.add_argument(
        "--vector-index-path",
        type=Path,
        help="Optional vector index to evaluate. When --bootstrap-docs is used, defaults to a retrieval-mode-specific file under storage/eval.",
    )
    parser.add_argument(
        "--bootstrap-docs",
        nargs="*",
        type=Path,
        default=[],
        help="Optional local docs to index before evaluation, for example README.md docs/architecture.md.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON path for detailed results.")
    parser.add_argument(
        "--fail-on-failures",
        action="store_true",
        help="Exit with status 1 when any evaluation example fails.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Force an offline local-fallback baseline even when .env contains an LLM key.",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail instead of accepting local fallback when retrieved context should reach the configured LLM.",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Run the frozen W8-T1 full RAG evaluation and create reproducible artifacts.",
    )
    parser.add_argument(
        "--formal-unanswerable",
        action="store_true",
        help="Run the frozen W8-T2 unanswerable/control evaluation.",
    )
    parser.add_argument(
        "--w7-manifest",
        type=Path,
        default=DEFAULT_W7_MANIFEST,
        help="Frozen W7 manifest that defines the selected retrieval/reranking configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FORMAL_OUTPUT_DIR,
        help="Directory for independently identified formal run artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_EVALUATION_REPORT,
        help="Human-readable report generated from the formal artifact.",
    )
    parser.add_argument(
        "--unanswerable-manifest",
        type=Path,
        default=DEFAULT_UNANSWERABLE_MANIFEST,
        help="Frozen W8-T2 case, absence, control and rubric manifest.",
    )
    parser.add_argument(
        "--unanswerable-output-dir",
        type=Path,
        default=DEFAULT_UNANSWERABLE_OUTPUT_DIR,
        help="Directory for independently identified W8-T2 artifacts.",
    )
    parser.add_argument(
        "--unanswerable-report-path",
        type=Path,
        default=DEFAULT_UNANSWERABLE_REPORT,
        help="Human-readable W8-T2 unanswerable evaluation report.",
    )
    parser.add_argument("--run-id", help="Optional explicit formal run identity.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the selected W8 artifact/report paths.",
    )
    args = parser.parse_args()

    if args.formal and args.formal_unanswerable:
        parser.error("--formal and --formal-unanswerable are mutually exclusive")
    if args.disable_llm and args.require_llm:
        parser.error("--disable-llm and --require-llm cannot be used together")
    if args.disable_llm:
        disable_llm_for_evaluation()
    if (args.require_llm or args.formal or args.formal_unanswerable) and not is_llm_configured():
        parser.error("formal/required LLM evaluation needs a configured API key or local placeholder")

    llm_metadata = get_llm_runtime_metadata(
        resolve_model_identity=(
            args.require_llm or args.formal or args.formal_unanswerable
        ),
    )
    if args.formal_unanswerable:
        if args.output is not None:
            parser.error(
                "--formal-unanswerable uses --unanswerable-output-dir instead of --output"
            )
        try:
            return run_formal_unanswerable_evaluation(
                args,
                llm_metadata=llm_metadata,
            )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            parser.error(str(exc))
    if args.formal:
        if args.output is not None:
            parser.error("--formal uses --output-dir instead of the legacy --output path")
        try:
            return run_formal_evaluation(args, llm_metadata=llm_metadata)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            parser.error(str(exc))

    index_path = args.index_path
    vector_index_path = args.vector_index_path
    if args.bootstrap_docs and index_path is None:
        index_path = default_eval_index(args.retrieval_mode, args.dataset)
    if args.bootstrap_docs and vector_index_path is None:
        vector_index_path = default_eval_vector_index(args.retrieval_mode, args.dataset)
    if index_path is not None and not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path
    if vector_index_path is not None and not vector_index_path.is_absolute():
        vector_index_path = PROJECT_ROOT / vector_index_path

    if args.bootstrap_docs:
        indexed = bootstrap_documents(args.bootstrap_docs, index_path, vector_index_path)
        print(f"Indexed {len(indexed)} bootstrap document(s):")
        for item in indexed:
            print(f"- {item['filename']} ({item['chunk_count']} chunks)")

    examples = load_dataset(args.dataset)
    results = [
        evaluate_example(
            example,
            args.top_k,
            index_path,
            vector_index_path,
            args.retrieval_mode,
            args.min_score,
            args.require_llm,
            user_id=args.user_id,
        )
        for example in examples
    ]
    summary = summarize(results)

    print("\nRAG evaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failed = [result for result in results if not result["passed"]]
    if failed:
        print("\nFailed examples")
        for result in failed:
            print(
                f"- [{result['category']}] {result['question']} "
                f"(sources={result['retrieved_sources']}, scores={result['retrieval_scores']}, "
                f"mode={result['answer_mode']})"
            )

    if args.output:
        output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "artifact_version": 1,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "configuration": {
                        "dataset": str(args.dataset),
                        "bootstrap_documents": [str(path) for path in args.bootstrap_docs],
                        "retrieval_mode": args.retrieval_mode,
                        "top_k": args.top_k,
                        "requested_min_score": args.min_score,
                        "effective_min_score": (
                            results[0]["min_score"] if results else args.min_score
                        ),
                        "embedding_provider": settings.embedding_provider,
                        "embedding_model": (
                            settings.local_embedding_model
                            if settings.embedding_provider == "local_model"
                            else settings.embedding_model
                        ),
                        "vector_store_backend": settings.vector_store_backend,
                        "require_llm": args.require_llm,
                        "llm": llm_metadata,
                    },
                    "summary": summary,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote detailed results to {output_path}")

    if args.fail_on_failures and summary["pass_rate"] < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
