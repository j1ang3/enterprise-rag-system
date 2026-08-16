import argparse
import json
import sys
from pathlib import Path
from typing import List
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag import (
    bootstrap_documents,
    default_eval_index,
    default_eval_vector_index,
    evaluate_example,
    load_dataset,
    summarize,
)


DEFAULT_BUSINESS_DATASET = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
DEFAULT_BUSINESS_DOCS = [
    PROJECT_ROOT / "eval_docs" / "hr_policy.md",
    PROJECT_ROOT / "eval_docs" / "expense_policy.md",
    PROJECT_ROOT / "eval_docs" / "security_policy.md",
    PROJECT_ROOT / "eval_docs" / "product_faq.md",
]
DEFAULT_THRESHOLDS = [
    0.0,
    0.03,
    0.05,
    0.08,
    0.1,
    0.12,
    0.15,
    0.18,
    0.2,
    0.22,
    0.25,
    0.3,
]


def _resolve_paths(paths: List[Path]) -> List[Path]:
    return [path if path.is_absolute() else PROJECT_ROOT / path for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan retrieval min_score thresholds.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_BUSINESS_DATASET)
    parser.add_argument("--user-id", type=UUID, required=True)
    parser.add_argument(
        "--bootstrap-docs",
        nargs="*",
        type=Path,
        default=DEFAULT_BUSINESS_DOCS,
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["keyword", "vector", "hybrid", "rerank"],
        default="keyword",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset_path = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    docs = _resolve_paths(args.bootstrap_docs)
    index_path = default_eval_index(f"{args.retrieval_mode}_threshold_scan")
    vector_index_path = default_eval_vector_index(f"{args.retrieval_mode}_threshold_scan")

    indexed = bootstrap_documents(docs, index_path, vector_index_path)
    examples = load_dataset(dataset_path)

    print(f"Indexed {len(indexed)} bootstrap document(s).")
    print(f"Scanning {len(args.thresholds)} threshold(s) for {args.retrieval_mode} retrieval.")

    scan_results = []
    for threshold in args.thresholds:
        results = [
            evaluate_example(
                example,
                args.top_k,
                index_path,
                vector_index_path,
                args.retrieval_mode,
                threshold,
                user_id=args.user_id,
            )
            for example in examples
        ]
        summary = summarize(results)
        scan_results.append(
            {
                "retrieval_mode": args.retrieval_mode,
                "min_score": threshold,
                **summary,
            }
        )

    print("\nThreshold scan summary")
    print(json.dumps(scan_results, ensure_ascii=False, indent=2))

    best = max(
        scan_results,
        key=lambda item: (
            item["pass_rate"],
            item["no_answer_accuracy"],
            item["retrieval_recall_at_k"],
            item["answer_keyword_accuracy"],
        ),
    )
    print("\nBest threshold")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    if args.output:
        output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"best": best, "results": scan_results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote threshold scan to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
