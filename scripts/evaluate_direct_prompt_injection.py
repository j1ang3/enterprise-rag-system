import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping
from unittest.mock import patch
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.config import settings
from app.auth.tokens import create_access_token
from app.evaluation.dataset import EvaluationCase
from app.evaluation.direct_prompt_injection import (
    DirectPromptInjectionRunner,
    classify_direct_prompt_injection_result,
    load_direct_prompt_injection_cases,
    load_direct_prompt_injection_manifest,
    render_direct_prompt_injection_report,
)
from app.evaluation.rag import RAGEvaluationRunner, write_artifact
from app.evaluation.unanswerable import (
    load_unanswerable_cases,
    load_unanswerable_manifest,
    verify_absence_against_corpus,
)
from app.services.llm_client import get_llm_runtime_metadata, is_llm_configured
from app.services.prompts import SYSTEM_PROMPT
from app.services.rag_service import answer_question
from scripts.evaluate_rag import _repository_state, _resolve_formal_inputs


DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "security" / "direct_prompt_injection_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "security" / "direct_prompt_injection_runs"
DEFAULT_REPORT = PROJECT_ROOT / "evals/results/generated_reports/direct-prompt-injection-report.md"
DEFAULT_DATASET = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
DEFAULT_W7_MANIFEST = PROJECT_ROOT / "evals" / "retrieval_evaluation_config.json"


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"w9-t2-{timestamp}-qwen3-8b"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_manifest_and_inputs(
    args: argparse.Namespace,
    *,
    llm_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    bundle = load_direct_prompt_injection_manifest(args.manifest, project_root=PROJECT_ROOT)
    manifest = bundle["manifest"]
    resolved = _resolve_formal_inputs(args, llm_metadata=llm_metadata)

    if bundle["stable_dataset_path"] != resolved["dataset"].path:
        raise ValueError("W9-T2 stable dataset differs from the frozen formal input")
    if bundle["w7_frozen_manifest_path"] != resolved["manifest_path"]:
        raise ValueError("W9-T2 W7 configuration differs from the frozen formal input")
    prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if prompt_sha != manifest["system_prompt_identity"]["sha256"]:
        raise ValueError("production system prompt drifted after the W9-T2 rubric freeze")

    policy = manifest["formal_model_policy"]
    identity = llm_metadata.get("model_identity")
    digest = identity.get("digest") if isinstance(identity, Mapping) else None
    if (
        llm_metadata.get("provider") != policy["provider"]
        or llm_metadata.get("model") != policy["model"]
        or digest != policy["expected_digest"]
    ):
        raise ValueError("formal model/provider/digest differs from the frozen W9-T2 policy")

    log_path = settings.rag_structured_log_path
    resolved_log = log_path if log_path.is_absolute() else PROJECT_ROOT / log_path
    if not settings.rag_structured_logging_enabled or "w9-t2" not in resolved_log.name.casefold():
        raise ValueError("formal W9-T2 requires an enabled, dedicated w9-t2 structured log path")
    if resolved_log.exists() and not args.overwrite:
        raise FileExistsError(f"dedicated W9-T2 log already exists: {resolved_log}")

    attacks = load_direct_prompt_injection_cases(bundle["case_file_path"])
    expected_ids = manifest["case_file"]["attack_ids"]
    if [case.attack_id for case in attacks] != expected_ids:
        raise ValueError("direct injection attack order or identity drifted")
    actual_categories = Counter(case.category for case in attacks)
    if dict(actual_categories) != manifest["case_file"]["category_counts"]:
        raise ValueError("direct injection category counts drifted")

    stable_dataset = resolved["dataset"]
    stable_by_id = {case.query_id: case for case in stable_dataset.cases}
    unanswerable_bundle = load_unanswerable_manifest(
        bundle["unanswerable_manifest_path"], project_root=PROJECT_ROOT
    )
    if unanswerable_bundle["case_file_path"] != bundle["unanswerable_case_file_path"]:
        raise ValueError("W9-T2 unanswerable case identity differs from W8-T2")
    unanswerable_cases = load_unanswerable_cases(
        bundle["unanswerable_case_file_path"], stable_dataset=stable_dataset
    )
    unanswerable_by_id = {case.query_id: case for case in unanswerable_cases}
    absence = verify_absence_against_corpus(
        unanswerable_cases,
        document_paths=unanswerable_bundle["document_paths"],
        chunk_index_path=unanswerable_bundle["chunk_index_path"],
    )

    base_cases: Dict[str, EvaluationCase] = dict(stable_by_id)
    base_cases.update(
        {query_id: case.to_evaluation_case() for query_id, case in unanswerable_by_id.items()}
    )
    for attack in attacks:
        if attack.base_case_id not in base_cases:
            raise ValueError(f"attack base case is missing: {attack.base_case_id}")
        if attack.base_source == "stable_dataset" and attack.base_case_id not in stable_by_id:
            raise ValueError(f"attack base source mismatch: {attack.attack_id}")
        if attack.base_source == "unanswerable_cases" and attack.base_case_id not in unanswerable_by_id:
            raise ValueError(f"attack base source mismatch: {attack.attack_id}")

    controls_spec = manifest["benign_controls"]
    controls = []
    for query_id in controls_spec["stable_answerable_query_ids"]:
        case = stable_by_id.get(query_id)
        if case is None or not case.answerable:
            raise ValueError(f"invalid answerable control: {query_id}")
        controls.append(case)
    for query_id in controls_spec["supplemental_unanswerable_query_ids"]:
        case = unanswerable_by_id.get(query_id)
        if case is None:
            raise ValueError(f"invalid unanswerable control: {query_id}")
        controls.append(case.to_evaluation_case())

    return {
        "bundle": bundle,
        "manifest": manifest,
        "resolved": resolved,
        "attacks": attacks,
        "base_cases": base_cases,
        "controls": controls,
        "absence_verification": {
            query_id: absence[query_id]
            for query_id in controls_spec["supplemental_unanswerable_query_ids"]
        },
        "structured_log_path": resolved_log.resolve(),
    }


def _endpoint_acceptance(
    *,
    attack_case: Any,
    configuration: Any,
    rubric: Mapping[str, Any],
    user_id: UUID,
) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from app.main import app

    def frozen_answer(
        question: str,
        _top_k: int,
        *,
        retrieval_mode: str,
        min_score: float | None,
        request_id: str,
        user_id: UUID,
    ) -> Dict[str, Any]:
        result = answer_question(
            question,
            configuration.top_k,
            retrieval_mode=configuration.retrieval_mode,
            min_score=None,
            index_path=configuration.index_path,
            vector_index_path=configuration.vector_index_path,
            reranked_hybrid_config=configuration.reranked_hybrid,
            request_id=request_id,
            user_id=user_id,
        )
        return {**result, "retrieval_mode": retrieval_mode, "min_score": min_score or 0.0}

    with patch("app.routers.chat.answer_question", side_effect=frozen_answer):
        response = TestClient(app).post(
            "/chat/",
            headers={"Authorization": f"Bearer {create_access_token(user_id).value}"},
            json={
                "question": attack_case.user_query,
                "top_k": configuration.top_k,
                "retrieval_mode": "rerank",
                "min_score": 0.0,
            },
        )
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if response.status_code == 200 and isinstance(data, Mapping):
        wrapped = {
            "status": "success",
            "actual": {
                "answer": data.get("answer"),
                "answer_mode": data.get("answer_mode"),
                "model": data.get("model"),
                "citations": data.get("citations", []),
                "contexts": data.get("contexts", []),
            },
        }
        security = classify_direct_prompt_injection_result(
            wrapped, attack_case=attack_case, rubric=rubric, formal=True
        )
        return {
            "status": "completed",
            "attack_id": attack_case.attack_id,
            "http_status": response.status_code,
            "request_id": data.get("request_id"),
            "qwen_reached": data.get("answer_mode") == "llm" and data.get("model") == "qwen3:8b",
            "security_outcome": security["outcome"],
            "security_evaluation": security,
            "answer": data.get("answer"),
            "answer_mode": data.get("answer_mode"),
            "model": data.get("model"),
            "citation_count": len(data.get("citations", [])),
            "context_count": len(data.get("contexts", [])),
            "configuration_note": "FastAPI Router plus production RAG; a test-only adapter injects the same frozen internal hybrid_rerank paths because that mode is not public API input.",
            "included_in_asr": False,
        }
    return {
        "status": "failed",
        "attack_id": attack_case.attack_id,
        "http_status": response.status_code,
        "qwen_reached": False,
        "security_outcome": "execution_failure",
        "response_detail": payload,
        "included_in_asr": False,
    }


def run(args: argparse.Namespace) -> int:
    if not is_llm_configured():
        raise ValueError("formal W9-T2 requires a configured local Ollama placeholder key")
    llm_metadata = get_llm_runtime_metadata(resolve_model_identity=True)
    prepared = _validate_manifest_and_inputs(args, llm_metadata=llm_metadata)
    manifest = prepared["manifest"]
    resolved = prepared["resolved"]
    configuration = resolved["configuration"]

    run_id = args.run_id or _run_id()
    artifact_path = args.output_dir.resolve() / f"{run_id}.json"
    report_path = args.report_path.resolve()
    if artifact_path.exists() and not args.overwrite:
        raise FileExistsError(f"direct injection artifact already exists: {artifact_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"direct injection report already exists: {report_path}")

    endpoint = _endpoint_acceptance(
        attack_case=prepared["attacks"][0],
        configuration=configuration,
        rubric=manifest["rubric"],
        user_id=args.user_id,
    )
    run_metadata = {
        "command": [sys.executable, *sys.argv],
        "repository": _repository_state(),
        "prompt_identity": manifest["system_prompt_identity"],
        "model_policy": manifest["formal_model_policy"],
        "structured_log": {
            "path": _relative(prepared["structured_log_path"]),
            "dedicated_to_task": True,
        },
        "randomness": {
            "configured_temperature": llm_metadata.get("temperature"),
            "seed": None,
            "guarantee": "configuration identity is reproducible; generations are not guaranteed bit-for-bit",
        },
        "classification": {
            "rubric_version": manifest["rubric"]["version"],
            "llm_judge": False,
            "manual_review_policy": manifest["rubric"]["manual_review_policy"],
        },
    }
    source_identities = {
        "w9_t2_manifest": {
            "path": _relative(prepared["bundle"]["manifest_path"]),
            "sha256": prepared["bundle"]["manifest_sha256"],
            "experiment_id": manifest["experiment_id"],
        },
        "attack_case_file": manifest["case_file"],
        "threat_model": manifest["threat_model"],
        "stable_dataset": resolved["dataset"].identity(project_root=PROJECT_ROOT),
        "w7_frozen_manifest": {
            "path": _relative(resolved["manifest_path"]),
            "sha256": resolved["validated"]["manifest_sha256"],
            "experiment_id": resolved["manifest"]["experiment_id"],
        },
        "corpus": resolved["corpus_identity"],
        "unanswerable_sources": {
            "case_file": manifest["unanswerable_case_file"],
            "manifest": manifest["unanswerable_manifest"],
            "absence_verification": prepared["absence_verification"],
        },
        "synthetic_canary": manifest["synthetic_canary"],
    }
    artifact = DirectPromptInjectionRunner(
        RAGEvaluationRunner(configuration),
        rubric=manifest["rubric"],
        formal=True,
    ).run(
        attack_cases=prepared["attacks"],
        base_cases=prepared["base_cases"],
        control_cases=prepared["controls"],
        run_id=run_id,
        run_metadata=run_metadata,
        source_identities=source_identities,
        endpoint_acceptance=endpoint,
    )
    write_artifact(artifact, artifact_path, overwrite=args.overwrite)
    report = render_direct_prompt_injection_report(
        artifact, artifact_path=_relative(artifact_path)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\nFormal W9-T2 direct prompt injection summary")
    print(json.dumps(artifact["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nEndpoint acceptance: {json.dumps(endpoint, ensure_ascii=False)}")
    print(f"Wrote artifact to {artifact_path}")
    print(f"Wrote report to {report_path}")
    failures = artifact["aggregate"]["case_counts"]["execution_failure"]
    endpoint_failed = not endpoint.get("qwen_reached")
    control_failed = any(row.get("status") != "success" for row in artifact["benign_controls"])
    return 1 if failures or endpoint_failed or control_failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen W9-T2 direct prompt injection baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--w7-manifest", type=Path, default=DEFAULT_W7_MANIFEST)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--retrieval-mode", default="hybrid_rerank")
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--vector-index-path", type=Path)
    parser.add_argument("--bootstrap-docs", nargs="*", type=Path, default=[])
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Existing PostgreSQL user authorized for the frozen evaluation corpus.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
