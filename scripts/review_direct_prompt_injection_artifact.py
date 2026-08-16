import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.evaluation.direct_prompt_injection import (
    aggregate_direct_prompt_injection_results,
    classify_direct_prompt_injection_result,
    file_sha256,
    load_direct_prompt_injection_cases,
    load_direct_prompt_injection_manifest,
    render_direct_prompt_injection_report,
)
from app.evaluation.rag import write_artifact


DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "security" / "direct_prompt_injection_config.json"
DEFAULT_REPORT = PROJECT_ROOT / "evals/results/generated_reports/direct-prompt-injection-report.md"


def review_artifact(
    source_path: Path,
    *,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"source execution artifact not found: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("task") != "W9-T2" or source.get("formal") is not True:
        raise ValueError("source must be a formal W9-T2 execution artifact")
    llm = source.get("resolved_configuration", {}).get("llm", {})
    if llm.get("provider") != "ollama" or llm.get("model") != "qwen3:8b":
        raise ValueError("source execution artifact is not Ollama qwen3:8b")

    bundle = load_direct_prompt_injection_manifest(manifest_path, project_root=PROJECT_ROOT)
    manifest = bundle["manifest"]
    cases = {
        case.attack_id: case
        for case in load_direct_prompt_injection_cases(bundle["case_file_path"])
    }
    reviewed = copy.deepcopy(source)
    for row in reviewed.get("attacks", []):
        attack_record = row.get("attack_case")
        attack_id = attack_record.get("attack_id") if isinstance(attack_record, Mapping) else None
        attack_case = cases.get(str(attack_id))
        if attack_case is None:
            raise ValueError(f"source artifact contains unknown attack: {attack_id}")
        row["security_evaluation"] = classify_direct_prompt_injection_result(
            row,
            attack_case=attack_case,
            rubric=manifest["rubric"],
            formal=True,
        )

    reviewed["aggregate"] = aggregate_direct_prompt_injection_results(reviewed["attacks"])
    reviewed["source_execution_artifact"] = {
        "path": _relative(source_path),
        "sha256": file_sha256(source_path),
        "run_id": source["run_id"],
    }
    reviewed["run_id"] = f"{source['run_id']}-reviewed"
    reviewed["generated_at"] = datetime.now(timezone.utc).isoformat()
    reviewed["classification_review"] = {
        "status": "completed",
        "rubric_version": manifest["rubric"]["version"],
        "llm_rerun": False,
        "reason": "Reapplied the unchanged pre-frozen success definitions after fixing punctuation normalization and negated-target handling in the deterministic classifier.",
        "source_outputs_modified": False,
        "manual_semantic_review": {
            "required_attack_ids": [
                row["attack_case"]["attack_id"]
                for row in reviewed["attacks"]
                if row["security_evaluation"]["outcome"] == "ambiguous"
            ],
            "result": "No ambiguous cases in the selected final execution artifact."
            if not any(
                row["security_evaluation"]["outcome"] == "ambiguous"
                for row in reviewed["attacks"]
            )
            else "Ambiguous cases remain conservatively unpromoted pending case-level semantic review.",
        },
    }
    write_artifact(reviewed, output_path.resolve(), overwrite=overwrite)
    report_path = report_path.resolve()
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"direct injection report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_direct_prompt_injection_report(
            reviewed, artifact_path=_relative(output_path)
        ),
        encoding="utf-8",
    )
    return reviewed


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reapply the frozen W9-T2 rubric to a saved formal execution artifact without rerunning the LLM."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.source.with_name(f"{args.source.stem}-reviewed.json")
    try:
        artifact = review_artifact(
            args.source,
            manifest_path=args.manifest,
            output_path=output,
            report_path=args.report_path,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(artifact["aggregate"], ensure_ascii=False, indent=2))
    print(f"Reviewed artifact: {output.resolve()}")
    print(f"Report: {args.report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
