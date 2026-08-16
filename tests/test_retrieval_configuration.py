import json
import tempfile
import unittest
from pathlib import Path

from app.evaluation.retrieval_configuration import (
    RetrievalExperimentConfig,
    build_candidate_count_matrix,
    build_final_top_k_matrix,
    run_configured_pipeline,
    select_candidate_configuration,
    select_final_top_k_configuration,
    validate_one_variable_matrix,
)
from scripts.evaluate_rag import load_dataset
from scripts.tune_retrieval_configuration import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    DEFAULT_SPLIT_MANIFEST,
    PROTECTED_W6_ARTIFACT,
    _file_sha256,
    aggregate_configuration,
    evaluate_configuration,
    select_tuning_examples,
    validate_split_manifest,
    write_artifact,
)


def _candidate(index: int) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": f"doc-{index}",
        "filename": f"doc-{index}.md",
        "content": f"candidate text {index}",
        "fused_score": 1.0 / index,
        "matched_sources": ["vector", "bm25"],
        "source_ranks": {"vector": index, "bm25": index},
        "source_scores": {"vector": 0.9, "bm25": 3.0},
        "metadata": {"created_at": "fixed", "chunk_index": index - 1},
    }


class RecordingHybridRetriever:
    def __init__(self, candidates=None):
        self.candidates = candidates or [_candidate(index) for index in range(1, 6)]
        self.calls = []

    def retrieve_fused(self, query, top_k, *, candidate_depth=None, rrf_k=60):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "candidate_depth": candidate_depth,
                "rrf_k": rrf_k,
            }
        )
        return [dict(candidate) for candidate in self.candidates[:top_k]]


class ReversingReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, *, top_k=None):
        self.calls.append(
            {"query": query, "candidate_count": len(candidates), "top_k": top_k}
        )
        reranked = [
            {**candidate, "rerank_score": float(rank)}
            for rank, candidate in enumerate(reversed(candidates), start=1)
        ]
        return reranked if top_k is None else reranked[:top_k]


class RetrievalConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = RetrievalExperimentConfig(
            per_source_candidate_depth=5,
            rrf_k=60,
            rerank_candidate_count=5,
            final_top_k=3,
        )

    def test_configuration_validates_stage_relationship(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            RetrievalExperimentConfig(0, 60, 5, 3)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            RetrievalExperimentConfig(5, 60, 2, 3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            RetrievalExperimentConfig(5, True, 5, 3)

    def test_config_id_and_serialization_are_stable(self):
        equivalent = RetrievalExperimentConfig(5, 60, 5, 3)
        self.assertEqual(self.baseline.config_id, "ps05-r060-n05-k03")
        self.assertEqual(self.baseline.config_id, equivalent.config_id)
        self.assertEqual(
            self.baseline.to_dict(),
            {
                "config_id": "ps05-r060-n05-k03",
                "per_source_candidate_depth": 5,
                "rrf_k": 60,
                "rerank_candidate_count": 5,
                "final_top_k": 3,
            },
        )

    def test_pipeline_propagates_each_parameter_to_the_correct_stage(self):
        retriever = RecordingHybridRetriever()
        reranker = ReversingReranker()

        run = run_configured_pipeline(
            "  policy query  ",
            self.baseline,
            hybrid_retriever=retriever,
            reranker=reranker,
        )

        self.assertEqual(
            retriever.calls,
            [
                {
                    "query": "policy query",
                    "top_k": 5,
                    "candidate_depth": 5,
                    "rrf_k": 60,
                }
            ],
        )
        self.assertEqual(
            reranker.calls,
            [{"query": "policy query", "candidate_count": 5, "top_k": 3}],
        )
        self.assertEqual(len(run.candidates_before_rerank), 5)
        self.assertEqual(len(run.results_after_rerank), 3)

    def test_final_cutoff_is_applied_after_reranking(self):
        retriever = RecordingHybridRetriever()
        reranker = ReversingReranker()

        run = run_configured_pipeline(
            "query",
            self.baseline,
            hybrid_retriever=retriever,
            reranker=reranker,
        )

        self.assertEqual(
            [item["chunk_id"] for item in run.candidates_before_rerank],
            ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
        )
        self.assertEqual(
            [item["chunk_id"] for item in run.results_after_rerank],
            ["chunk-5", "chunk-4", "chunk-3"],
        )

    def test_pipeline_rejects_backends_that_break_candidate_contracts(self):
        class TooManyRetriever(RecordingHybridRetriever):
            def retrieve_fused(self, query, top_k, *, candidate_depth=None, rrf_k=60):
                return [_candidate(index) for index in range(1, top_k + 2)]

        with self.assertRaisesRegex(RuntimeError, "more candidates"):
            run_configured_pipeline(
                "query",
                self.baseline,
                hybrid_retriever=TooManyRetriever(),
                reranker=ReversingReranker(),
            )

    def test_matrices_change_exactly_one_variable(self):
        candidate_configs = build_candidate_count_matrix(self.baseline, [5, 3, 8])
        self.assertEqual(
            [config.rerank_candidate_count for config in candidate_configs],
            [5, 3, 8],
        )
        self.assertEqual({config.final_top_k for config in candidate_configs}, {3})

        final_configs = build_final_top_k_matrix(candidate_configs[1], [3, 1, 2])
        self.assertEqual([config.final_top_k for config in final_configs], [3, 1, 2])
        self.assertEqual(
            {config.rerank_candidate_count for config in final_configs}, {3}
        )

        with self.assertRaisesRegex(ValueError, "must not contain duplicate"):
            build_candidate_count_matrix(self.baseline, [5, 5])
        with self.assertRaisesRegex(ValueError, "only final_top_k may vary"):
            validate_one_variable_matrix(
                [self.baseline, RetrievalExperimentConfig(6, 60, 5, 2)],
                variable="final_top_k",
            )

    def test_selection_rules_are_deterministic_and_prefer_smaller_ties(self):
        candidate_configs = build_candidate_count_matrix(self.baseline, [5, 3, 8])
        tied_metrics = {
            config.config_id: {
                "candidate_recall": 1.0,
                "final_hit_rate": 1.0,
                "final_recall": 1.0,
                "final_mrr": 1.0,
            }
            for config in candidate_configs
        }
        selected_candidate = select_candidate_configuration(
            candidate_configs,
            tied_metrics,
            baseline_config_id=self.baseline.config_id,
        )
        self.assertEqual(selected_candidate.rerank_candidate_count, 3)

        final_configs = build_final_top_k_matrix(selected_candidate, [3, 1, 2])
        final_metrics = {
            config.config_id: {
                "candidate_recall": 1.0,
                "final_hit_rate": 1.0,
                "final_recall": 1.0,
                "final_mrr": 1.0,
            }
            for config in final_configs
        }
        selected_final = select_final_top_k_configuration(
            final_configs,
            final_metrics,
            baseline_config_id=final_configs[0].config_id,
        )
        self.assertEqual(selected_final.final_top_k, 2)

    def test_split_is_disjoint_complete_and_keeps_q027_held_out(self):
        examples = load_dataset(DEFAULT_DATASET)
        manifest = json.loads(DEFAULT_SPLIT_MANIFEST.read_text(encoding="utf-8"))
        groups = validate_split_manifest(
            manifest,
            examples,
            dataset_sha256=_file_sha256(DEFAULT_DATASET),
        )

        self.assertEqual(len(groups["tuning_query_ids"]), 18)
        self.assertEqual(len(groups["heldout_query_ids"]), 5)
        self.assertEqual(len(groups["unanswerable_query_ids"]), 4)
        self.assertIn("q027", groups["heldout_query_ids"])
        selected = select_tuning_examples(examples, groups["tuning_query_ids"])
        self.assertEqual([query_id for query_id, _ in selected], groups["tuning_query_ids"])
        self.assertNotIn("q027", [query_id for query_id, _ in selected])

    def test_split_rejects_dataset_hash_drift(self):
        examples = load_dataset(DEFAULT_DATASET)
        manifest = json.loads(DEFAULT_SPLIT_MANIFEST.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            validate_split_manifest(manifest, examples, dataset_sha256="changed")

    def test_per_query_output_and_aggregate_contain_required_diagnostics(self):
        config = RetrievalExperimentConfig(5, 60, 3, 2)
        example = {
            "question": "Which policy?",
            "expected_sources": ["doc-3.md"],
            "expected_citation_chunk_ids": ["chunk-3"],
            "category": "test",
            "difficulty": "easy",
        }
        results = evaluate_configuration(
            config,
            [("q001", example)],
            hybrid_retriever=RecordingHybridRetriever(),
            reranker=ReversingReranker(),
        )
        aggregate = aggregate_configuration(results)

        self.assertEqual(results[0]["actual_reranker_input_count"], 3)
        self.assertEqual(results[0]["actual_final_result_count"], 2)
        self.assertEqual(
            results[0]["relevant_document_ranks"]["before"], {"doc-3.md": [3]}
        )
        self.assertEqual(
            results[0]["relevant_document_ranks"]["after"], {"doc-3.md": [1]}
        )
        self.assertIn("rerank_score", results[0]["after_rerank"][0])
        self.assertEqual(results[0]["configuration_id"], config.config_id)
        self.assertEqual(
            results[0]["after_rerank"][0]["matched_sources"], ["vector", "bm25"]
        )
        self.assertEqual(aggregate["candidate_recall"], 1.0)
        self.assertEqual(aggregate["final_recall"], 1.0)
        self.assertEqual(aggregate["strict_chunk_diagnostic"]["query_ids"], ["q001"])

    def test_artifact_write_preserves_existing_files_and_protects_w6(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            output.write_text('{"existing": true}', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_artifact(output, {"replacement": True})
            self.assertEqual(output.read_text(encoding="utf-8"), '{"existing": true}')

            write_artifact(output, {"replacement": True}, overwrite=True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"replacement": True})

        with self.assertRaisesRegex(ValueError, "W6-T4"):
            write_artifact(PROTECTED_W6_ARTIFACT, {"unsafe": True}, overwrite=True)

    def test_real_result_artifact_contains_required_identity_and_metrics(self):
        payload = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(payload["task"], "W7-T2")
        self.assertIn("sha256", payload["dataset_identity"])
        self.assertEqual(
            payload["split_identity"]["executed_query_ids"],
            payload["split_identity"]["tuning_query_ids"],
        )
        self.assertFalse(payload["split_identity"]["heldout_results_present"])
        self.assertIn("config_id", payload["selected_configuration_for_w7_t3"])
        for config_id, results in payload["per_query_results"].items():
            self.assertTrue(results)
            self.assertEqual(results[0]["configuration_id"], config_id)
            self.assertIn("candidate_metrics", results[0])
            self.assertIn("final_metrics", results[0])


if __name__ == "__main__":
    unittest.main()
