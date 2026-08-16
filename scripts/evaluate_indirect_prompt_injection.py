import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.config import settings
from app.auth.tokens import create_access_token
from app.evaluation.indirect_prompt_injection import (
    IndirectPromptInjectionCase,
    TracedReranker,
    build_artifact,
    build_delivery_evidence,
    classify_model_outcome,
    compact_rag_result,
    evaluate_clean_control,
    execution_failure,
    file_sha256,
    final_classification,
    ingest_fixture_corpus,
    load_indirect_prompt_injection_cases,
    load_indirect_prompt_injection_manifest,
    render_indirect_prompt_injection_report,
    utc_run_id,
    validate_fixture_safety,
)
from app.retrieval.reranker import get_default_reranker
from app.services.knowledge_base import index_document as production_index_document
from app.services.llm_client import get_llm_runtime_metadata, is_llm_configured
from app.services.prompts import SYSTEM_PROMPT
from app.services.rag_service import answer_question
from app.services.search_service import RerankedHybridConfig
from app.services.storage_paths import DocumentStoragePaths
from scripts.evaluate_rag import _repository_state


DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "security" / "indirect_prompt_injection_config.json"
DEFAULT_CASES = PROJECT_ROOT / "evals" / "security" / "indirect_prompt_injection_cases.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "security" / "indirect_prompt_injection_runs"
DEFAULT_REPORT = PROJECT_ROOT / "evals/results/generated_reports/indirect-prompt-injection-report.md"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / "storage" / "security" / "w9_t3"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {_relative(path): file_sha256(path) for path in paths}


def _validate_preflight(
    args: argparse.Namespace, *, llm_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    bundle = load_indirect_prompt_injection_manifest(args.manifest, project_root=PROJECT_ROOT)
    manifest = bundle["manifest"]
    cases = load_indirect_prompt_injection_cases(args.cases)
    if file_sha256(args.cases) != manifest["attack_dataset"]["sha256"]:
        raise ValueError("W9-T3 attack dataset drifted after freeze")
    if [case.attack_id for case in cases] != manifest["attack_dataset"]["attack_ids"]:
        raise ValueError("W9-T3 attack order or identity drifted")
    if dict(Counter(case.category for case in cases)) != manifest["attack_dataset"]["category_counts"]:
        raise ValueError("W9-T3 category distribution drifted")
    safety = validate_fixture_safety(cases, project_root=PROJECT_ROOT)

    prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if prompt_sha != manifest["system_under_test"]["system_prompt_sha256"]:
        raise ValueError("production system prompt drifted after W9-T3 freeze")
    policy = manifest["formal_model_policy"]
    identity = llm_metadata.get("model_identity")
    digest = identity.get("digest") if isinstance(identity, Mapping) else None
    if (
        llm_metadata.get("provider") != policy["provider"]
        or llm_metadata.get("model") != policy["model"]
        or digest != policy["expected_digest"]
    ):
        raise ValueError("formal model/provider/digest differs from W9-T3 freeze")
    if llm_metadata.get("temperature") != manifest["system_under_test"]["temperature"]:
        raise ValueError("formal generation temperature drifted")
    if llm_metadata.get("max_tokens") != manifest["system_under_test"]["max_tokens"]:
        raise ValueError("formal generation max_tokens drifted")

    config = RerankedHybridConfig(**manifest["system_under_test"]["reranked_hybrid"])
    log_path = settings.rag_structured_log_path
    log_path = log_path if log_path.is_absolute() else PROJECT_ROOT / log_path
    if not settings.rag_structured_logging_enabled or "w9-t3" not in log_path.name.casefold():
        raise ValueError("formal W9-T3 requires an enabled dedicated w9-t3 log path")
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(f"dedicated W9-T3 log already exists: {log_path}")
    return {
        "bundle": bundle,
        "manifest": manifest,
        "cases": cases,
        "safety": safety,
        "configuration": config,
        "log_path": log_path.resolve(),
    }


def _execute_case(
    case: IndirectPromptInjectionCase,
    *,
    chunk_index_path: Path,
    vector_index_path: Path,
    configuration: RerankedHybridConfig,
    user_id: UUID,
) -> dict[str, Any]:
    traced = TracedReranker(get_default_reranker())
    try:
        result = answer_question(
            case.user_query,
            configuration.final_top_k,
            retrieval_mode="hybrid_rerank",
            min_score=None,
            index_path=chunk_index_path,
            vector_index_path=vector_index_path,
            reranked_hybrid_config=configuration,
            reranker=traced,
            user_id=user_id,
        )
        return compact_rag_result(result, full_ranking=traced.full_ranking)
    except Exception as exc:
        return execution_failure(exc)


def _corpus_identity(root: Path, chunk_path: Path, vector_path: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "root": _relative(root),
        "chunk_index_path": _relative(chunk_path),
        "vector_index_path": _relative(vector_path),
        "file_hashes": _hash_paths(files),
        "normal_corpus": False,
        "task_isolated": True,
    }


def _run_endpoint_acceptance(
    case: IndirectPromptInjectionCase,
    *,
    root: Path,
    configuration: RerankedHybridConfig,
    formal: bool,
    user_id: UUID,
) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from app.main import app

    upload_dir = root / "uploads"
    text_dir = root / "texts"
    index_dir = root / "index"
    chunk_path = index_dir / "chunks.json"
    vector_path = index_dir / "vectors.json"
    storage_paths = DocumentStoragePaths(
        upload_dir=upload_dir,
        text_dir=text_dir,
        index_dir=index_dir,
        chunks_file=chunk_path,
        vectors_file=vector_path,
    )
    captured_chunks: list[dict[str, Any]] = []

    def isolated_index(document_id: str, filename: str, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        chunks = production_index_document(
            document_id,
            filename,
            text,
            index_path=chunk_path,
            vector_index_path=vector_path,
            sections=kwargs.get("sections"),
        )
        captured_chunks.extend(chunks)
        return chunks

    fixture = case.fixture_path(PROJECT_ROOT, "malicious")
    with patch("app.routers.documents.get_document_storage_paths", return_value=storage_paths), patch(
        "app.routers.documents.index_document", side_effect=isolated_index
    ):
        response = TestClient(app).post(
            "/documents/upload",
            headers={"Authorization": f"Bearer {create_access_token(user_id).value}"},
            files={"file": (fixture.name, fixture.read_bytes(), "text/plain")},
        )
    payload = response.json()
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if response.status_code != 200 or not isinstance(data, Mapping):
        return {
            "status": "execution_failure",
            "http_status": response.status_code,
            "response_detail": payload,
            "included_in_metrics": False,
        }
    marker_ids = [chunk["chunk_id"] for chunk in captured_chunks if case.document_marker in chunk["content"]]
    ingestion = {
        "document_ingested": True,
        "malicious_chunk_created": bool(marker_ids),
        "malicious_chunk_ids": marker_ids,
        "document_id": data.get("document_id"),
        "chunk_ids": [chunk["chunk_id"] for chunk in captured_chunks],
    }
    execution = _execute_case(
        case,
        chunk_index_path=chunk_path,
        vector_index_path=vector_path,
        configuration=configuration,
        user_id=user_id,
    )
    delivery = build_delivery_evidence(execution, ingestion=ingestion)
    model = classify_model_outcome(execution, case=case, delivery=delivery, formal=formal)
    return {
        "status": "completed" if model["outcome"] != "execution_failure" else "execution_failure",
        "http_status": response.status_code,
        "attack_id": case.attack_id,
        "document_id": data.get("document_id"),
        "saved_path": _relative(Path(str(data.get("saved_path")))),
        "text_path": _relative(Path(str(data.get("text_path")))),
        "chunk_count": data.get("chunk_count"),
        "production_upload_router_used": True,
        "production_extractor_chunker_vector_index_used": True,
        "qwen_reached": execution.get("answer_mode") == "llm" and execution.get("model") == "qwen3:8b",
        "execution": execution,
        "delivery_evidence": delivery,
        "model_evaluation": model,
        "final_classification": final_classification(delivery, model),
        "included_in_metrics": False,
        "isolation_note": "A test-only storage adapter routes the production upload/index components to this W9-T3-only directory.",
    }


def run(args: argparse.Namespace) -> int:
    if not is_llm_configured():
        raise ValueError("formal W9-T3 requires a configured local Ollama placeholder key")
    llm_metadata = get_llm_runtime_metadata(resolve_model_identity=True)
    prepared = _validate_preflight(args, llm_metadata=llm_metadata)
    manifest = prepared["manifest"]
    cases = prepared["cases"]
    configuration = prepared["configuration"]
    run_id = args.run_id or utc_run_id()
    run_root = args.storage_root.resolve() / run_id
    artifact_path = args.output_dir.resolve() / f"{run_id}.json"
    report_path = args.report_path.resolve()
    if run_root.exists() or artifact_path.exists():
        raise FileExistsError("W9-T3 run identity already exists; choose a new run id")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"W9-T3 report already exists: {report_path}")

    historical_paths = [
        (PROJECT_ROOT / identity["path"]).resolve()
        for identity in manifest["historical_immutability"]
    ]
    historical_before = _hash_paths(historical_paths)
    clean_root = run_root / "clean"
    attack_root = run_root / "attack"
    clean_chunks = clean_root / "chunks.json"
    clean_vectors = clean_root / "vectors.json"
    attack_chunks = attack_root / "chunks.json"
    attack_vectors = attack_root / "vectors.json"
    clean_ingestion = ingest_fixture_corpus(
        cases,
        variant="clean",
        project_root=PROJECT_ROOT,
        chunk_index_path=clean_chunks,
        vector_index_path=clean_vectors,
    )
    attack_ingestion = ingest_fixture_corpus(
        cases,
        variant="malicious",
        project_root=PROJECT_ROOT,
        chunk_index_path=attack_chunks,
        vector_index_path=attack_vectors,
    )
    clean_by_id = {row["attack_id"]: row for row in clean_ingestion}
    attack_by_id = {row["attack_id"]: row for row in attack_ingestion}

    controls = []
    for case in cases:
        execution = _execute_case(
            case,
            chunk_index_path=clean_chunks,
            vector_index_path=clean_vectors,
            configuration=configuration,
            user_id=args.user_id,
        )
        controls.append(
            {
                "attack_id": case.attack_id,
                "pair_id": case.pair_id,
                "query": case.user_query,
                "ingestion": clean_by_id[case.attack_id],
                "execution": execution,
                "control_evaluation": evaluate_clean_control(execution, case=case, formal=True),
            }
        )

    attacks = []
    for case in cases:
        execution = _execute_case(
            case,
            chunk_index_path=attack_chunks,
            vector_index_path=attack_vectors,
            configuration=configuration,
            user_id=args.user_id,
        )
        delivery = build_delivery_evidence(execution, ingestion=attack_by_id[case.attack_id])
        model = classify_model_outcome(execution, case=case, delivery=delivery, formal=True)
        attacks.append(
            {
                "attack_case": case.to_dict(),
                "ingestion": attack_by_id[case.attack_id],
                "execution": execution,
                "delivery_evidence": delivery,
                "model_evaluation": model,
                "final_classification": final_classification(delivery, model),
            }
        )

    endpoint = _run_endpoint_acceptance(
        cases[0],
        root=run_root / "endpoint_acceptance",
        configuration=configuration,
        formal=True,
        user_id=args.user_id,
    )
    corpus_identities = {
        "clean": _corpus_identity(clean_root, clean_chunks, clean_vectors),
        "attack": _corpus_identity(attack_root, attack_chunks, attack_vectors),
        "endpoint_acceptance": _corpus_identity(
            run_root / "endpoint_acceptance",
            run_root / "endpoint_acceptance" / "index" / "chunks.json",
            run_root / "endpoint_acceptance" / "index" / "vectors.json",
        ),
    }
    historical_after = _hash_paths(historical_paths)
    if historical_after != historical_before:
        raise RuntimeError("historical W8/W9 evidence changed during W9-T3")

    fixtures = [
        {"path": _relative(case.fixture_path(PROJECT_ROOT, variant)), "sha256": file_sha256(case.fixture_path(PROJECT_ROOT, variant))}
        for case in cases
        for variant in ("clean", "malicious")
    ]
    source_identities = {
        "manifest": {
            "path": _relative(prepared["bundle"]["manifest_path"]),
            "sha256": prepared["bundle"]["manifest_sha256"],
            "experiment_id": manifest["experiment_id"],
        },
        "attack_dataset": manifest["attack_dataset"],
        "fixtures": fixtures,
        "threat_model": manifest["source_identities"]["threat_model"],
        "normal_corpus_before_after_sha256": historical_after,
    }
    artifact = build_artifact(
        run_id=run_id,
        run_metadata={
            "command": [sys.executable, *sys.argv],
            "repository": _repository_state(),
            "llm": dict(llm_metadata),
            "system_under_test": manifest["system_under_test"],
            "structured_log": {"path": _relative(prepared["log_path"]), "dedicated_to_task": True},
            "fixture_safety": prepared["safety"],
            "paired_experiment": True,
            "authorization_user_id": str(args.user_id),
            "controlled_delivery_subtest": "not_used",
            "no_attack_score_boosting": True,
            "no_defense_implementation": True,
        },
        source_identities=source_identities,
        corpus_identities=corpus_identities,
        ingestion_records=[*clean_ingestion, *attack_ingestion],
        attacks=attacks,
        clean_controls=controls,
        endpoint_acceptance=endpoint,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_indirect_prompt_injection_report(artifact, artifact_path=_relative(artifact_path)),
        encoding="utf-8",
    )

    print("\nFormal W9-T3 indirect prompt injection summary")
    print(json.dumps(artifact["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nEndpoint acceptance: {json.dumps(endpoint, ensure_ascii=False)}")
    print(f"Wrote artifact to {artifact_path}")
    print(f"Wrote report to {report_path}")
    execution_failures = artifact["aggregate"]["model_outcome_counts"]["execution_failure"]
    control_failures = sum(row["control_evaluation"]["status"] == "execution_failure" for row in controls)
    endpoint_failed = not endpoint.get("qwen_reached")
    return 1 if execution_failures or control_failures or endpoint_failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen W9-T3 indirect prompt-injection baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL user authorized for the isolated fixture corpus.",
    )
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
