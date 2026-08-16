import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
BOOTSTRAP_DOCS = [
    "eval_docs/hr_policy.md",
    "eval_docs/expense_policy.md",
    "eval_docs/security_policy.md",
    "eval_docs/product_faq.md",
]


REGRESSION_EVALS = [
    {
        "name": "business-vector",
        "dataset": "evals/business_policy_eval.jsonl",
        "retrieval_mode": "vector",
        "output": "storage/eval/business_vector_baseline.json",
        "min_pass_rate": 0.8889,
    },
    {
        "name": "hard-hybrid",
        "dataset": "evals/business_policy_hard_eval.jsonl",
        "retrieval_mode": "hybrid",
        "output": "storage/eval/business_hard_hybrid_expanded.json",
        "min_pass_rate": 1.0,
    },
    {
        "name": "hard-rerank",
        "dataset": "evals/business_policy_hard_eval.jsonl",
        "retrieval_mode": "rerank",
        "output": "storage/eval/business_hard_rerank_expanded.json",
        "min_pass_rate": 1.0,
    },
]


def _run(command: List[str]) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def _run_unit_tests() -> bool:
    return _run([PYTHON, "-m", "unittest", "discover", "-s", "tests"]) == 0


def _run_eval(spec: Dict[str, object]) -> bool:
    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    run_dir = PROJECT_ROOT / "storage" / "regression" / f"{spec['name']}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "results.json"
    index_path = run_dir / "chunks.json"
    vector_index_path = run_dir / "vectors.json"
    command = [
        PYTHON,
        "scripts/evaluate_rag.py",
        "--dataset",
        str(spec["dataset"]),
        "--retrieval-mode",
        str(spec["retrieval_mode"]),
        "--disable-llm",
        "--index-path",
        str(index_path),
        "--vector-index-path",
        str(vector_index_path),
        "--bootstrap-docs",
        *BOOTSTRAP_DOCS,
        "--output",
        str(output),
    ]
    if _run(command) != 0:
        return False

    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        summary = payload["summary"]
        pass_rate = float(summary["pass_rate"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        print(f"Could not read evaluation summary: {output}")
        return False

    min_pass_rate = float(spec["min_pass_rate"])
    if pass_rate < min_pass_rate:
        print(
            f"{spec['name']} failed threshold: pass_rate={pass_rate} "
            f"< min_pass_rate={min_pass_rate}"
        )
        return False

    print(
        f"{spec['name']} passed threshold: pass_rate={pass_rate} "
        f">= min_pass_rate={min_pass_rate}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local RAG regression checks.")
    parser.add_argument(
        "--skip-unit-tests",
        action="store_true",
        help="Skip vector-store backend unit tests.",
    )
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="Skip RAG evaluation baselines.",
    )
    args = parser.parse_args()

    checks = []
    if not args.skip_unit_tests:
        checks.append(("unit-tests", _run_unit_tests()))

    if not args.skip_evals:
        for spec in REGRESSION_EVALS:
            checks.append((str(spec["name"]), _run_eval(spec)))

    failed = [name for name, passed in checks if not passed]
    if failed:
        print("\nRegression failed:")
        for name in failed:
            print(f"- {name}")
        return 1

    print("\nRegression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
