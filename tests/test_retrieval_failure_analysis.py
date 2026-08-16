import copy
import unittest
from pathlib import Path

from scripts.analyze_retrieval_failures import (
    ArtifactValidationError,
    analyze_artifact,
    build_outcome_groups,
    classify_rank_movement,
    is_partial_multi_relevant,
    load_artifact,
    validate_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_ARTIFACT = (
    PROJECT_ROOT / "evals" / "results" / "W6-T4-retrieval-evaluation.json"
)
METHODS = ("vector", "bm25", "hybrid_rrf")


def _metrics(recall: float, reciprocal_rank: float) -> dict:
    first_rank = int(round(1 / reciprocal_rank)) if reciprocal_rank else None
    return {
        str(k): {
            "k": k,
            "hit": reciprocal_rank > 0,
            "recall": recall,
            "reciprocal_rank": reciprocal_rank,
            "first_relevant_rank": first_rank,
            "matched_relevant_ids": ["policy.md"] if recall else [],
        }
        for k in (1, 3, 5)
    }


def _result(
    query_id: str,
    method: str,
    *,
    recall: float = 1.0,
    reciprocal_rank: float = 1.0,
    relevant_documents: list[str] | None = None,
) -> dict:
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "method": method,
        "category": "policy",
        "difficulty": "easy",
        "should_answer": True,
        "retrieved_chunk_ids": [f"{method}-{query_id}"],
        "retrieved_document_labels": ["policy.md"],
        "retrieval_scores": [1.0],
        "relevant_document_labels": relevant_documents or ["policy.md"],
        "relevant_chunk_ids": None,
        "relevance_label_type": "document_filename",
        "metrics_status": "evaluated",
        "metrics_by_k": _metrics(recall, reciprocal_rank),
        "latency_ms": 1.0,
    }


def _artifact(rows_by_method: dict[str, list[dict]]) -> dict:
    query_count = len(rows_by_method["vector"])
    summary_metrics = {
        str(k): {"hit_rate": 1.0, "recall": 1.0, "mrr": 1.0}
        for k in (1, 3, 5)
    }
    empty_comparison = {
        "bm25_better_than_vector": [],
        "vector_better_than_bm25": [],
        "hybrid_better_than_vector": [],
        "hybrid_worse_than_vector": [],
        "all_methods_missed": [],
    }
    return {
        "task": "W6-T4",
        "generated_at": "2026-08-03T00:00:00+00:00",
        "ground_truth": {
            "source": "evals/test.jsonl",
            "label_type": "document_filename",
            "strict_chunk_label_coverage": 0,
            "dataset_sha256": "dataset-hash",
        },
        "configuration": {
            "query_count": query_count,
            "methods": list(METHODS),
            "k_values": [1, 3, 5],
            "hybrid_candidate_depth": 5,
            "rrf_k": 60,
            "indexed_chunk_count": 3,
            "indexed_documents": [{"filename": "policy.md", "chunk_count": 3}],
            "bootstrap_document_sha256": {"eval_docs/policy.md": "corpus-hash"},
        },
        "summary": {
            method: {"metrics_by_k": copy.deepcopy(summary_metrics)}
            for method in METHODS
        },
        "comparisons_by_k": {
            str(k): copy.deepcopy(empty_comparison) for k in (1, 3, 5)
        },
        "results": rows_by_method,
    }


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid = _artifact(
            {method: [_result("q001", method)] for method in METHODS}
        )

    def test_missing_method_is_rejected(self):
        artifact = copy.deepcopy(self.valid)
        artifact["configuration"]["methods"] = ["vector", "bm25"]

        with self.assertRaisesRegex(ArtifactValidationError, "methods"):
            validate_artifact(artifact)

    def test_mismatched_query_ids_are_rejected(self):
        artifact = copy.deepcopy(self.valid)
        artifact["results"]["bm25"][0]["query_id"] = "q999"

        with self.assertRaisesRegex(ArtifactValidationError, "identical query IDs"):
            validate_artifact(artifact)

    def test_missing_metrics_are_rejected(self):
        artifact = copy.deepcopy(self.valid)
        del artifact["results"]["vector"][0]["metrics_by_k"]["3"]

        with self.assertRaisesRegex(ArtifactValidationError, "metrics at K=3"):
            validate_artifact(artifact)

    def test_missing_source_metadata_is_rejected(self):
        artifact = copy.deepcopy(self.valid)
        del artifact["ground_truth"]["dataset_sha256"]

        with self.assertRaisesRegex(ArtifactValidationError, "dataset_sha256"):
            validate_artifact(artifact)

    def test_missing_artifact_version_is_a_warning_not_fabricated(self):
        warnings = validate_artifact(self.valid)

        self.assertEqual(
            warnings,
            ["W6-T4 source artifact does not declare artifact_version"],
        )


class OutcomeGroupingTests(unittest.TestCase):
    def test_all_pairwise_outcomes_and_equal_are_grouped(self):
        qualities = {
            "q001": {"vector": 0.5, "bm25": 1.0, "hybrid_rrf": 1.0},
            "q002": {"vector": 1.0, "bm25": 0.5, "hybrid_rrf": 0.5},
            "q003": {"vector": 0.5, "bm25": 0.5, "hybrid_rrf": 1.0},
            "q004": {"vector": 1.0, "bm25": 1.0, "hybrid_rrf": 0.5},
            "q005": {"vector": 1.0, "bm25": 1.0, "hybrid_rrf": 1.0},
        }
        rows = {
            method: [
                _result(query_id, method, recall=method_values[method])
                for query_id, method_values in qualities.items()
            ]
            for method in METHODS
        }

        groups = build_outcome_groups(_artifact(rows))["3"]["groups"]

        self.assertEqual(groups["bm25_better_than_vector"], ["q001"])
        self.assertEqual(groups["vector_better_than_bm25"], ["q002"])
        self.assertEqual(groups["hybrid_better_than_vector"], ["q001", "q003"])
        self.assertEqual(groups["hybrid_worse_than_vector"], ["q002", "q004"])
        self.assertEqual(groups["hybrid_equal_to_vector"], ["q005"])
        self.assertEqual(groups["bm25_better_than_hybrid"], ["q004"])
        self.assertEqual(groups["hybrid_better_than_bm25"], ["q003"])

    def test_empty_outcome_group_is_preserved(self):
        artifact = _artifact(
            {method: [_result("q001", method)] for method in METHODS}
        )

        grouped = build_outcome_groups(artifact)["3"]

        self.assertEqual(grouped["groups"]["hybrid_better_than_vector"], [])
        self.assertIn("hybrid_better_than_vector", grouped["empty_groups"])


class RankMovementTests(unittest.TestCase):
    def test_rank_improvement_crosses_cutoff(self):
        movement = classify_rank_movement(5, 3, k=3)

        self.assertEqual(movement["direction"], "improved")
        self.assertEqual(movement["cutoff_effect"], "crossed_into_top_k")

    def test_rank_improvement_without_crossing_cutoff(self):
        movement = classify_rank_movement(5, 4, k=3)

        self.assertEqual(movement["direction"], "improved")
        self.assertEqual(
            movement["cutoff_effect"],
            "moved_up_but_did_not_cross_cutoff",
        )

    def test_rank_decline_is_recorded(self):
        movement = classify_rank_movement(2, 4, k=3)

        self.assertEqual(movement["direction"], "declined")
        self.assertEqual(movement["cutoff_effect"], "crossed_out_of_top_k")

    def test_unretrieved_item_is_insufficient_evidence(self):
        movement = classify_rank_movement(None, None, k=3)

        self.assertEqual(movement["direction"], "not_observed_in_either")
        self.assertEqual(
            movement["cutoff_effect"],
            "insufficient_evidence_outside_observed_depth",
        )

    def test_partial_multi_relevant_can_have_hit_and_high_mrr(self):
        result = _result(
            "q001",
            "vector",
            recall=0.5,
            reciprocal_rank=1.0,
            relevant_documents=["hr.md", "security.md"],
        )

        self.assertTrue(result["metrics_by_k"]["3"]["hit"])
        self.assertEqual(result["metrics_by_k"]["3"]["reciprocal_rank"], 1.0)
        self.assertTrue(is_partial_multi_relevant(result, k=3))


class RealArtifactAnalysisTests(unittest.TestCase):
    def test_real_artifact_validates_and_preserves_q027_evidence(self):
        artifact = load_artifact(REAL_ARTIFACT)
        warnings = validate_artifact(artifact, project_root=PROJECT_ROOT)
        analysis = analyze_artifact(
            artifact,
            source_artifact=str(REAL_ARTIFACT.relative_to(PROJECT_ROOT)),
            validation_warnings=warnings,
        )

        self.assertEqual(
            analysis["observed_findings"]["bm25_strict_advantage_by_k"],
            {"1": [], "3": ["q027"], "5": []},
        )
        q027 = analysis["representative_cases"]["q027"]
        self.assertEqual(
            q027["methods"]["vector"]["strict_relevant_chunk_ranks"][
                "eval-hr-policy-2"
            ],
            5,
        )
        self.assertEqual(
            q027["methods"]["hybrid_rrf"]["strict_relevant_chunk_ranks"][
                "eval-hr-policy-2"
            ],
            4,
        )


if __name__ == "__main__":
    unittest.main()
