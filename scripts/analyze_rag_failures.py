"""Create the offline W8-T4 derived artifact and human-readable report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.failure_analysis import (  # noqa: E402
    build_failure_analysis_artifact,
    file_sha256,
    load_failure_analysis_manifest,
    load_json_object,
    load_jsonl_events,
    render_failure_analysis_report,
    write_json_artifact,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "evals" / "failure_analysis_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "failure_analysis_runs"
DEFAULT_REPORT = PROJECT_ROOT / "evals/results/generated_reports/failure-analysis-report.md"


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"w8-t4-{timestamp}-qwen3-8b"


def _repository_state() -> Dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "worktree_dirty": bool(status) if status is not None else None,
    }


def run_analysis(
    *,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    overwrite_report: bool = False,
) -> Mapping[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"failure analysis artifact already exists: {output_path}")
    if report_path.exists() and not overwrite_report:
        raise FileExistsError(f"failure analysis report already exists: {report_path}")

    loaded = load_failure_analysis_manifest(
        manifest_path,
        project_root=PROJECT_ROOT,
    )
    manifest = loaded["manifest"]
    evaluation_path = loaded["evaluation_path"]
    unanswerable_path = loaded["unanswerable_path"]
    runtime_paths = loaded["runtime_paths"]
    source_hashes_before = {
        evaluation_path: file_sha256(evaluation_path),
        unanswerable_path: file_sha256(unanswerable_path),
        **{path: file_sha256(path) for path in runtime_paths},
    }

    evaluation = load_json_object(evaluation_path, label="W8-T1 source artifact")
    unanswerable = load_json_object(
        unanswerable_path, label="W8-T2 source artifact"
    )
    runtime_events = []
    runtime_identities = []
    for record, path in zip(manifest["source_runtime_logs"], runtime_paths):
        events = load_jsonl_events(path)
        runtime_events.extend(events)
        runtime_identities.append(
            {
                **dict(record),
                "event_count": len(events),
                "request_ids": [event.get("request_id") for event in events],
                "statuses": [event.get("status") for event in events],
            }
        )

    run_id = output_path.stem
    artifact = build_failure_analysis_artifact(
        evaluation=evaluation,
        unanswerable=unanswerable,
        manifest=manifest,
        manifest_identity={
            "path": str(manifest_path.resolve().relative_to(PROJECT_ROOT)),
            "sha256": loaded["manifest_sha256"],
        },
        runtime_events=runtime_events,
        runtime_identities=runtime_identities,
        run_id=run_id,
        repository_state=_repository_state(),
    )
    write_json_artifact(artifact, output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        recorded_artifact_path = str(output_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        recorded_artifact_path = str(output_path.resolve())
    report_path.write_text(
        render_failure_analysis_report(
            artifact,
            artifact_path=recorded_artifact_path,
        ),
        encoding="utf-8",
    )

    source_hashes_after = {
        path: file_sha256(path) for path in source_hashes_before
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("a historical W8 source artifact or log changed during analysis")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze frozen W8 RAG evidence without rerunning qwen3:8b."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Allow replacing only the derived human report, never source artifacts.",
    )
    args = parser.parse_args()
    run_id = _new_run_id()
    output = args.output or DEFAULT_OUTPUT_DIR / f"{run_id}.json"
    artifact = run_analysis(
        manifest_path=args.manifest,
        output_path=output,
        report_path=args.report,
        overwrite_report=args.overwrite_report,
    )
    print(
        json.dumps(
            {
                "run_id": artifact["run_id"],
                "artifact": str(output),
                "report": str(args.report),
                "quality_failures": artifact["aggregate"][
                    "quality_failure_case_count"
                ],
                "needs_review": artifact["aggregate"]["needs_review_count"],
                "targeted_reproductions": artifact["selection"][
                    "targeted_reproduction_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
