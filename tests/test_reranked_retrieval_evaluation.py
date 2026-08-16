import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.evaluation.retrieval_comparison import (
    METHOD_NAMES,
    build_pairwise_comparisons,
    build_reranker_effect,
    validate_comparable_results,
)
from scripts.evaluate_reranked_retrieval import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    PROTECTED_ARTIFACTS,
    _build_result_row,
    evaluate_query_methods,
    validate_frozen_manifest,
    write_artifact,
)


def _candidate(index: int, filename: str) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": f"doc-{index}",
        "filename": filename,
        "content": f"candidate {index}",
        "position": index,
        "chunk_index": index - 1,
        "source": "rrf",
        "retrieval_mode": "rrf",
        "score": 1.0 / index,
        "fused_score": 1.0 / index,
        "matched_sources": ["vector", "bm25"],
        "source_ranks": {"vector": index, "bm25": index},
        "source_scores": {"vector": 0.9, "bm25": 3.0},
        "metadata": {"chunk_index": index - 1, "created_at": "fixed"},
    }


class RecordingRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.last_return = None

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        self.last_return = [deepcopy(result) for result in self.results[:top_k]]
        return self.last_return


class RecordingHybrid:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.last_return = None

    def retrieve_fused(self, query, top_k, *, candidate_depth=None, rrf_k=60):
        self.calls.append((query, top_k, candidate_depth, rrf_k))
        self.last_return = [deepcopy(result) for result in self.results[:top_k]]
        return self.last_return


class RecordingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, *, top_k=None):
        self.calls.append(
            {
                "query": query,
                "candidate_object": candidates,
                "candidate_ids": [candidate["chunk_id"] for candidate in candidates],
                "top_k": top_k,
            }
        )
        reranked = [
            {**deepcopy(candidate), "rerank_score": float(rank)}
            for rank, candidate in enumerate(reversed(candidates), start=1)
        ]
        return reranked if top_k is None else reranked[:top_k]


class RerankedRetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "offline_evaluation_depth": 3,
            "metric_k_values": [1, 2, 3],
            "rrf_output_count": 3,
            "per_source_candidate_depth": 5,
            "rrf_k": 60,
        }
        self.example = {
            "question": "Which policy applies?",
            "expected_sources": ["relevant.md"],
            "expected_citation_chunk_ids": ["chunk-3"],
            "category": "test",
            "difficulty": "hard",
            "should_answer": True,
        }

    def test_three_methods_are_registered_in_required_order(self):
        self.assertEqual(
            METHOD_NAMES, ("vector", "hybrid_rrf", "hybrid_reranker")
        )

    def test_method_calls_and_reranker_difference_are_isolated(self):
        vector = RecordingRetriever(
            [_candidate(3, "relevant.md"), _candidate(2, "other.md")]
        )
        hybrid = RecordingHybrid(
            [
                _candidate(1, "other.md"),
                _candidate(2, "other.md"),
                _candidate(3, "relevant.md"),
            ]
        )
        reranker = RecordingReranker()

        rows = evaluate_query_methods(
            "q001",
            self.example,
            vector_retriever=vector,
            hybrid_retriever=hybrid,
            reranker=reranker,
            config=self.config,
        )

        self.assertEqual(vector.calls, [("Which policy applies?", 3)])
        self.assertEqual(hybrid.calls, [("Which policy applies?", 3, 5, 60)])
        self.assertEqual(reranker.calls[0]["candidate_object"], hybrid.last_return)
        self.assertIs(reranker.calls[0]["candidate_object"], hybrid.last_return)
        self.assertEqual(reranker.calls[0]["candidate_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(reranker.calls[0]["top_k"], 3)
        self.assertEqual(rows["hybrid_rrf"]["retrieved_chunk_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(rows["hybrid_reranker"]["retrieved_chunk_ids"], ["chunk-3", "chunk-2", "chunk-1"])
        self.assertEqual(rows["hybrid_reranker"]["pre_rerank_chunk_ids"], rows["hybrid_rrf"]["retrieved_chunk_ids"])
        self.assertIsNone(rows["hybrid_rrf"]["ranked_results"][0]["rerank_score"])
        self.assertEqual(rows["hybrid_reranker"]["ranked_results"][0]["rerank_score"], 1.0)
        self.assertEqual(
            rows["hybrid_reranker"]["ranked_results"][0]["matched_sources"],
            ["vector", "bm25"],
        )
        self.assertEqual(
            rows["hybrid_reranker"]["ranked_results"][0]["metadata"][
                "created_at"
            ],
            "fixed",
        )

    def test_all_methods_preserve_same_query_ground_truth_and_k_values(self):
        rows = evaluate_query_methods(
            "q001",
            self.example,
            vector_retriever=RecordingRetriever([_candidate(3, "relevant.md")]),
            hybrid_retriever=RecordingHybrid([_candidate(3, "relevant.md")]),
            reranker=RecordingReranker(),
            config=self.config,
        )
        results_by_method = {method: [rows[method]] for method in METHOD_NAMES}

        query_ids = validate_comparable_results(
            results_by_method, k_values=[1, 2, 3]
        )

        self.assertEqual(query_ids, ["q001"])
        for method in METHOD_NAMES:
            self.assertEqual(results_by_method[method][0]["query"], self.example["question"])
            self.assertEqual(results_by_method[method][0]["relevant_document_labels"], ["relevant.md"])
            self.assertEqual(set(results_by_method[method][0]["metrics_by_k"]), {"1", "2", "3"})

    def test_comparison_rejects_query_ground_truth_and_k_drift(self):
        rows = self._synthetic_results(
            vector_qualities=[(1.0, 1.0)],
            hybrid_qualities=[(1.0, 1.0)],
            reranker_qualities=[(1.0, 1.0)],
        )
        rows["hybrid_rrf"][0]["query_id"] = "different"
        with self.assertRaisesRegex(ValueError, "same ordered query IDs"):
            validate_comparable_results(rows, k_values=[1])

        rows = self._synthetic_results(
            vector_qualities=[(1.0, 1.0)],
            hybrid_qualities=[(1.0, 1.0)],
            reranker_qualities=[(1.0, 1.0)],
        )
        rows["hybrid_rrf"][0]["relevant_document_labels"] = ["changed.md"]
        with self.assertRaisesRegex(ValueError, "relevance labels differ"):
            validate_comparable_results(rows, k_values=[1])

        rows = self._synthetic_results(
            vector_qualities=[(1.0, 1.0)],
            hybrid_qualities=[(1.0, 1.0)],
            reranker_qualities=[(1.0, 1.0)],
        )
        del rows["hybrid_rrf"][0]["metrics_by_k"]["1"]
        with self.assertRaisesRegex(ValueError, "same metric K"):
            validate_comparable_results(rows, k_values=[1])

    def test_missing_labels_cannot_enter_formal_comparison(self):
        rows = self._synthetic_results(
            vector_qualities=[(1.0, 1.0)],
            hybrid_qualities=[(1.0, 1.0)],
            reranker_qualities=[(1.0, 1.0)],
        )
        rows["vector"][0]["relevant_document_labels"] = []
        with self.assertRaisesRegex(ValueError, "require document relevance"):
            validate_comparable_results(rows, k_values=[1])

    def test_pairwise_groups_better_worse_equal_and_empty_groups(self):
        rows = self._synthetic_results(
            vector_qualities=[(0.0, 0.0), (1.0, 1.0), (1.0, 1.0)],
            hybrid_qualities=[(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)],
            reranker_qualities=[(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)],
        )

        comparisons = build_pairwise_comparisons(rows, k_values=[1])
        groups = comparisons["comparisons"]["vector_vs_hybrid"]["by_k"]["1"]

        self.assertEqual([item["query_id"] for item in groups["challenger_better"]], ["q001"])
        self.assertEqual([item["query_id"] for item in groups["challenger_worse"]], ["q002"])
        self.assertEqual([item["query_id"] for item in groups["equal"]], ["q003"])
        reranker_groups = comparisons["comparisons"]["hybrid_vs_hybrid_reranker"]["by_k"]["1"]
        self.assertEqual(reranker_groups["empty_groups"], ["challenger_better", "challenger_worse"])

    def test_rank_movement_records_promotion_cutoff_and_candidate_failure(self):
        example = {
            **self.example,
            "expected_sources": ["relevant.md", "missing.md"],
        }
        pre = [
            _candidate(1, "other.md"),
            _candidate(3, "relevant.md"),
            _candidate(2, "other.md"),
        ]
        post = [pre[1], pre[0], pre[2]]
        hybrid = _build_result_row(
            "q001",
            example,
            method="hybrid_rrf",
            ranked_results=pre,
            k_values=[1, 3],
            latency_ms=1.0,
            stage_latency_ms={},
        )
        reranked = _build_result_row(
            "q001",
            example,
            method="hybrid_reranker",
            ranked_results=post,
            k_values=[1, 3],
            latency_ms=2.0,
            stage_latency_ms={},
            pre_rerank_results=pre,
        )

        effect = build_reranker_effect([hybrid], [reranked], k_values=[1, 3])
        movements = effect["items"][0]["document_level_movements"]

        self.assertEqual(movements[0]["rank_delta"], 1)
        self.assertEqual(movements[0]["movement_by_k"]["1"]["direction"], "improved")
        self.assertEqual(movements[0]["movement_by_k"]["1"]["cutoff_effect"], "crossed_into_top_k")
        self.assertFalse(movements[1]["candidate_retrieved"])
        self.assertTrue(effect["items"][0]["candidate_recall_failure_document_level"])
        self.assertEqual(effect["summary"]["document_relevant_item_direction_counts"], {"promoted": 1, "candidate_not_retrieved": 1})
        self.assertEqual(effect["summary"]["rank_changed_but_metric_unchanged_query_ids_by_k"]["3"], [])

    def test_rank_changed_but_metrics_unchanged_is_reported(self):
        pre = [
            _candidate(1, "relevant.md"),
            _candidate(2, "other.md"),
            _candidate(3, "another.md"),
        ]
        post = [pre[0], pre[2], pre[1]]
        hybrid = _build_result_row(
            "q001",
            self.example,
            method="hybrid_rrf",
            ranked_results=pre,
            k_values=[1, 3],
            latency_ms=1.0,
            stage_latency_ms={},
        )
        reranked = _build_result_row(
            "q001",
            self.example,
            method="hybrid_reranker",
            ranked_results=post,
            k_values=[1, 3],
            latency_ms=2.0,
            stage_latency_ms={},
            pre_rerank_results=pre,
        )

        effect = build_reranker_effect([hybrid], [reranked], k_values=[1, 3])

        self.assertEqual(
            effect["summary"]["rank_changed_but_metric_unchanged_query_ids_by_k"],
            {"1": ["q001"], "3": ["q001"]},
        )

    def test_reranker_effect_rejects_different_candidate_sets(self):
        pre = [_candidate(1, "relevant.md"), _candidate(2, "other.md")]
        changed = [_candidate(1, "relevant.md"), _candidate(3, "other.md")]
        hybrid = _build_result_row(
            "q001",
            self.example,
            method="hybrid_rrf",
            ranked_results=pre,
            k_values=[1],
            latency_ms=1.0,
            stage_latency_ms={},
        )
        reranked = _build_result_row(
            "q001",
            self.example,
            method="hybrid_reranker",
            ranked_results=changed,
            k_values=[1],
            latency_ms=2.0,
            stage_latency_ms={},
            pre_rerank_results=pre,
        )
        with self.assertRaisesRegex(ValueError, "permutation"):
            build_reranker_effect([hybrid], [reranked], k_values=[1])

    def test_real_frozen_manifest_matches_w7_t2_and_excludes_unanswerable(self):
        # Other test modules exercise mutable process-level settings. Pin this
        # test to the frozen runtime so its result does not depend on test order.
        with (
            patch.object(settings, "embedding_provider", "local_model"),
            patch.object(
                settings,
                "local_embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            patch.object(settings, "vector_store_backend", "faiss"),
            patch.object(
                settings,
                "reranker_model_name",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            patch.object(settings, "reranker_local_files_only", True),
        ):
            resolved = validate_frozen_manifest(DEFAULT_MANIFEST)
        self.assertEqual(resolved["evaluation_query_ids"], ["q005", "q010", "q015", "q020", "q027"])
        self.assertEqual(resolved["config"]["selected_w7_t2_config_id"], "ps05-r060-n03-k02")
        self.assertEqual(resolved["config"]["metric_k_values"], [1, 2, 3])
        self.assertTrue(set(resolved["evaluation_query_ids"]).isdisjoint(resolved["split"]["unanswerable_query_ids"]))

    def test_manifest_rejects_configuration_drift_before_evaluation(self):
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        manifest["resolved_configuration"]["reranker_candidate_count"] = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from W7-T2"):
                validate_frozen_manifest(path)

    def test_manifest_rejects_unanswerable_query_in_primary_set(self):
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        manifest["split"]["evaluation_query_ids"] = ["q021"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "heldout IDs"):
                validate_frozen_manifest(path)

    def test_artifact_write_preserves_existing_and_protected_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text('{"existing": true}', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_artifact(output, {"replacement": True})
            self.assertEqual(output.read_text(encoding="utf-8"), '{"existing": true}')

        protected = next(iter(PROTECTED_ARTIFACTS))
        with self.assertRaisesRegex(ValueError, "protected"):
            write_artifact(protected, {"unsafe": True}, overwrite=True)

    def test_real_artifact_has_frozen_identity_methods_and_shared_candidates(self):
        artifact = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        expected_ids = ["q005", "q010", "q015", "q020", "q027"]

        self.assertEqual(artifact["task"], "W7-T3")
        self.assertTrue(
            artifact["frozen_manifest"]["created_before_heldout_results"]
        )
        self.assertEqual(
            artifact["dataset_identity"]["evaluation_query_ids"], expected_ids
        )
        self.assertEqual(artifact["resolved_configuration"]["metric_k_values"], [1, 2, 3])
        self.assertFalse(artifact["split_identity"]["tuning_results_included"])
        self.assertFalse(artifact["split_identity"]["unanswerable_results_included"])
        self.assertEqual(tuple(artifact["per_query_results"]), METHOD_NAMES)
        for method in METHOD_NAMES:
            self.assertEqual(
                [row["query_id"] for row in artifact["per_query_results"][method]],
                expected_ids,
            )
        for hybrid, reranked in zip(
            artifact["per_query_results"]["hybrid_rrf"],
            artifact["per_query_results"]["hybrid_reranker"],
            strict=True,
        ):
            self.assertEqual(
                hybrid["retrieved_chunk_ids"], reranked["pre_rerank_chunk_ids"]
            )
        self.assertIn("pairwise_comparisons", artifact)
        self.assertIn("reranker_effect", artifact)

    @staticmethod
    def _synthetic_results(
        *,
        vector_qualities,
        hybrid_qualities,
        reranker_qualities,
    ):
        qualities_by_method = {
            "vector": vector_qualities,
            "hybrid_rrf": hybrid_qualities,
            "hybrid_reranker": reranker_qualities,
        }
        results = {method: [] for method in METHOD_NAMES}
        for method in METHOD_NAMES:
            for position, (recall, mrr) in enumerate(
                qualities_by_method[method], start=1
            ):
                results[method].append(
                    {
                        "query_id": f"q{position:03d}",
                        "query": f"query {position}",
                        "method": method,
                        "metrics_status": "evaluated",
                        "relevant_document_labels": ["relevant.md"],
                        "retrieved_chunk_ids": [f"{method}-{position}"],
                        "metrics_by_k": {
                            "1": {
                                "recall": recall,
                                "reciprocal_rank": mrr,
                                "first_relevant_rank": 1 if mrr else None,
                            }
                        },
                    }
                )
        return results


if __name__ == "__main__":
    unittest.main()
