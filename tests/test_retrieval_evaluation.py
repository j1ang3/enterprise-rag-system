import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.evaluation.retrieval import (
    aggregate_retrieval_results,
    calculate_retrieval_metrics,
)
from scripts.evaluate_retrieval import (
    RetrievalMethod,
    build_retrieval_methods,
    evaluate_method,
    validate_examples,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_hit_recall_and_mrr_use_ranked_results_at_k(self):
        metrics = calculate_retrieval_metrics(
            ["other.md", "relevant-a.md", "relevant-b.md"],
            ["relevant-a.md", "relevant-b.md"],
            top_k=2,
        )

        self.assertTrue(metrics["hit"])
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["first_relevant_rank"], 2)

    def test_duplicate_retrieved_identity_does_not_inflate_recall(self):
        metrics = calculate_retrieval_metrics(
            ["relevant-a.md", "relevant-a.md", "other.md"],
            ["relevant-a.md", "relevant-b.md"],
            top_k=3,
        )

        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["matched_relevant_ids"], ["relevant-a.md"])

    def test_empty_results_are_a_valid_miss(self):
        metrics = calculate_retrieval_metrics([], ["relevant.md"], top_k=3)

        self.assertFalse(metrics["hit"])
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.0)
        self.assertIsNone(metrics["first_relevant_rank"])

    def test_invalid_k_and_missing_relevance_labels_raise(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            calculate_retrieval_metrics(["a"], ["a"], top_k=0)
        with self.assertRaisesRegex(ValueError, "relevant_ids"):
            calculate_retrieval_metrics(["a"], [], top_k=1)

    def test_aggregate_uses_only_evaluated_queries_for_quality(self):
        evaluated = {
            "metrics_status": "evaluated",
            "latency_ms": 2.0,
            "metrics_by_k": {
                "1": {"hit": True, "recall": 0.5, "reciprocal_rank": 1.0}
            },
        }
        not_applicable = {
            "metrics_status": "not_applicable_no_relevant_documents",
            "latency_ms": 4.0,
            "metrics_by_k": {},
        }

        summary = aggregate_retrieval_results(
            [evaluated, not_applicable],
            k_values=[1],
        )

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["evaluated_query_count"], 1)
        self.assertEqual(summary["metrics_by_k"]["1"]["hit_rate"], 1.0)
        self.assertEqual(summary["metrics_by_k"]["1"]["recall"], 0.5)
        self.assertEqual(summary["latency_ms"]["mean"], 3.0)


class RetrievalEvaluationRunnerTests(unittest.TestCase):
    def test_dataset_validation_rejects_answerable_query_without_labels(self):
        with self.assertRaisesRegex(ValueError, "no document relevance labels"):
            validate_examples(
                [{"question": "answerable", "should_answer": True, "expected_sources": []}]
            )

    def test_evaluate_method_records_per_query_metrics_and_metadata(self):
        retriever = Mock(
            return_value=[
                {
                    "chunk_id": "chunk-1",
                    "filename": "policy.md",
                    "score": 0.9,
                }
            ]
        )
        method = RetrievalMethod("vector", retriever)
        examples = [
            {
                "question": "What is the policy?",
                "expected_sources": ["policy.md"],
                "expected_citation_chunk_ids": ["chunk-1"],
                "category": "policy",
                "difficulty": "easy",
                "should_answer": True,
            }
        ]

        results = evaluate_method(method, examples, k_values=[1, 3])

        retriever.assert_called_once_with("What is the policy?", 3)
        self.assertEqual(results[0]["query_id"], "q001")
        self.assertEqual(results[0]["retrieved_chunk_ids"], ["chunk-1"])
        self.assertEqual(results[0]["relevant_chunk_ids"], ["chunk-1"])
        self.assertTrue(results[0]["metrics_by_k"]["1"]["hit"])

    def test_build_methods_uses_vector_bm25_and_fused_hybrid_interfaces(self):
        vector = Mock(source="vector")
        vector.retrieve.return_value = []
        bm25 = Mock(source="bm25")
        bm25.retrieve.return_value = []
        hybrid = Mock()
        hybrid.retrieve_fused.return_value = []
        allowed_document_ids = frozenset({"eval-hr-policy"})

        with (
            patch("scripts.evaluate_retrieval.VectorRetriever", return_value=vector) as vector_cls,
            patch(
                "scripts.evaluate_retrieval.build_bm25_retriever",
                return_value=bm25,
            ) as bm25_builder,
            patch("scripts.evaluate_retrieval.HybridRetriever", return_value=hybrid) as hybrid_cls,
        ):
            methods = build_retrieval_methods(
                Path("chunks.json"),
                Path("vectors.json"),
                candidate_depth=5,
                rrf_k=60,
                allowed_document_ids=allowed_document_ids,
            )
            methods["vector"].retrieve("query", 5)
            methods["bm25"].retrieve("query", 5)
            methods["hybrid_rrf"].retrieve("query", 5)

        vector_cls.assert_called_once_with(
            index_path=Path("vectors.json"),
            min_score=0.0,
            allowed_document_ids=allowed_document_ids,
        )
        bm25_builder.assert_called_once_with(
            Path("chunks.json"),
            allowed_document_ids=allowed_document_ids,
        )
        hybrid_cls.assert_called_once_with(
            [vector, bm25],
            allowed_document_ids=allowed_document_ids,
        )
        vector.retrieve.assert_called_once_with("query", 5)
        bm25.retrieve.assert_called_once_with("query", 5)
        hybrid.retrieve_fused.assert_called_once_with(
            "query", 5, candidate_depth=5, rrf_k=60
        )


if __name__ == "__main__":
    unittest.main()
