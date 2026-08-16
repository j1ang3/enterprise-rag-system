import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services import vector_store


PYTHON = sys.executable
BOOTSTRAP_DOCS = [
    "eval_docs/hr_policy.md",
    "eval_docs/expense_policy.md",
    "eval_docs/security_policy.md",
    "eval_docs/product_faq.md",
]
QDRANT_EVALS = [
    {
        "name": "qdrant-business-vector",
        "dataset": "evals/business_policy_eval.jsonl",
        "retrieval_mode": "vector",
        "min_pass_rate": 0.9167,
    },
    {
        "name": "qdrant-hard-hybrid",
        "dataset": "evals/business_policy_hard_eval.jsonl",
        "retrieval_mode": "hybrid",
        "min_pass_rate": 1.0,
    },
    {
        "name": "qdrant-hard-rerank",
        "dataset": "evals/business_policy_hard_eval.jsonl",
        "retrieval_mode": "rerank",
        "min_pass_rate": 1.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG regression against a real Qdrant vector store.",
    )
    parser.add_argument("--url", default=settings.vector_store_url or "http://localhost:6333")
    parser.add_argument("--api-key", default=settings.vector_store_api_key)
    parser.add_argument(
        "--collection-prefix",
        default="enterprise_rag_regression",
        help="Prefix for temporary Qdrant collections created by this run.",
    )
    parser.add_argument(
        "--require-running",
        action="store_true",
        help="Fail instead of skipping when Qdrant is not reachable.",
    )
    parser.add_argument(
        "--keep-collections",
        action="store_true",
        help="Leave temporary Qdrant collections in place for inspection.",
    )
    return parser.parse_args()


def configure_qdrant(url: str, api_key: str, collection: str) -> None:
    settings.vector_store_backend = "external"
    settings.vector_store_external_provider = "qdrant"
    settings.vector_store_url = url
    settings.vector_store_api_key = api_key
    settings.vector_store_collection = collection


def qdrant_is_reachable(url: str, api_key: str) -> bool:
    configure_qdrant(url, api_key, "enterprise_rag_regression_probe")
    return vector_store._qdrant_request("GET", "/collections") is not None


def delete_collection(collection: str) -> None:
    settings.vector_store_collection = collection
    vector_store._qdrant_request("DELETE", vector_store._qdrant_collection_path())


def _run(command: List[str], env: Dict[str, str]) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return completed.returncode


def _run_eval(spec: Dict[str, object], args: argparse.Namespace, run_id: str) -> bool:
    collection = f"{args.collection_prefix}_{spec['retrieval_mode']}_{uuid4().hex[:10]}"
    run_dir = PROJECT_ROOT / "storage" / "qdrant-regression" / f"{spec['name']}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "results.json"
    index_path = run_dir / "chunks.json"
    vector_index_path = run_dir / "vectors.json"

    env = os.environ.copy()
    env.update(
        {
            "VECTOR_STORE_BACKEND": "external",
            "VECTOR_STORE_EXTERNAL_PROVIDER": "qdrant",
            "VECTOR_STORE_URL": args.url,
            "VECTOR_STORE_COLLECTION": collection,
            "VECTOR_STORE_API_KEY": args.api_key,
        }
    )

    command = [
        PYTHON,
        "scripts/evaluate_rag.py",
        "--dataset",
        str(spec["dataset"]),
        "--retrieval-mode",
        str(spec["retrieval_mode"]),
        "--index-path",
        str(index_path),
        "--vector-index-path",
        str(vector_index_path),
        "--bootstrap-docs",
        *BOOTSTRAP_DOCS,
        "--output",
        str(output),
    ]

    try:
        if _run(command, env) != 0:
            return False

        payload = json.loads(output.read_text(encoding="utf-8"))
        pass_rate = float(payload["summary"]["pass_rate"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        print(f"Could not read evaluation summary: {output}")
        return False
    finally:
        if not args.keep_collections:
            configure_qdrant(args.url, args.api_key, collection)
            delete_collection(collection)

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
    args = parse_args()
    print(f"Qdrant URL: {args.url}")

    if not qdrant_is_reachable(args.url, args.api_key):
        print("Qdrant regression skipped: Qdrant is not reachable.")
        return 1 if args.require_running else 0

    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    checks = [(str(spec["name"]), _run_eval(spec, args, run_id)) for spec in QDRANT_EVALS]

    failed = [name for name, passed in checks if not passed]
    if failed:
        print("\nQdrant regression failed:")
        for name in failed:
            print(f"- {name}")
        return 1

    print("\nQdrant regression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
