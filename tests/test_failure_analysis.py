import json
import tempfile
import unittest
from pathlib import Path

from app.evaluation.failure_analysis import (
    aggregate_classifications,
    build_failure_analysis_artifact,
    classify_failure_case,
    file_sha256,
    index_runtime_events,
    join_case_runtime_evidence,
    load_failure_analysis_manifest,
    load_json_object,
    load_jsonl_events,
    observed_w8_t1_signal_ids,
    observed_w8_t2_failure_ids,
    render_failure_analysis_report,
    validate_source_artifacts,
    write_json_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "evals" / "failure_analysis_config.json"


def answerable_evidence(**overrides):
    evidence = {
        "execution_status": "success",
        "answerable": True,
        "observed_signals": ["answer_bad"],
        "answer_quality": "unacceptable",
        "citation_quality": "acceptable",
        "expected_documents": ["policy.md"],
        "expected_chunk_ids": ["c1"],
        "expected_match": "all",
        "candidate_ids_by_field": {
            "chunk_id": ["c1"],
            "filename": ["policy.md"],
        },
        "final_ids_by_field": {
            "chunk_id": ["c1"],
            "filename": ["policy.md"],
        },
        "context_ids_by_field": {
            "chunk_id": ["c1"],
            "filename": ["policy.md"],
        },
    }
    evidence.update(overrides)
    return evidence


class FailureAnalysisTests(unittest.TestCase):
    def test_01_artifact_join_uses_run_query_and_request_identity(self):
        event = {
            "request_id": "11111111-1111-1111-1111-111111111111",
            "status": "success",
        }
        result = {
            "query_id": "q001",
            "actual": {"request_id": event["request_id"]},
        }
        joined = join_case_runtime_evidence(
            result,
            source_run_id="formal-run",
            runtime_events=index_runtime_events([event]),
        )
        self.assertEqual(joined["source_run_id"], "formal-run")
        self.assertEqual(joined["query_id"], "q001")
        self.assertEqual(joined["request_id"], event["request_id"])
        self.assertEqual(joined["join_status"], "joined_by_request_id")

    def test_02_retrieval_failure_when_required_candidate_is_absent(self):
        evidence = answerable_evidence(
            candidate_ids_by_field={
                "chunk_id": ["other"],
                "filename": ["other.md"],
            }
        )
        result = classify_failure_case(evidence)
        self.assertEqual(result["primary_failure"], "retrieval_failure")
        self.assertEqual(result["evidence_strength"], "confirmed")

    def test_03_ranking_failure_when_candidate_falls_below_cutoff(self):
        evidence = answerable_evidence(
            final_ids_by_field={
                "chunk_id": ["other"],
                "filename": ["other.md"],
            }
        )
        self.assertEqual(
            classify_failure_case(evidence)["primary_failure"],
            "ranking_failure",
        )

    def test_04_context_failure_when_final_evidence_is_not_in_context(self):
        evidence = answerable_evidence(
            context_ids_by_field={
                "chunk_id": ["other"],
                "filename": ["other.md"],
            }
        )
        self.assertEqual(
            classify_failure_case(evidence)["primary_failure"],
            "context_construction_failure",
        )

    def test_05_generation_failure_requires_sufficient_context(self):
        result = classify_failure_case(answerable_evidence())
        self.assertEqual(result["primary_failure"], "generation_failure")

    def test_06_correct_answer_with_wrong_citation_is_citation_failure(self):
        evidence = answerable_evidence(
            answer_quality="acceptable",
            citation_quality="incorrect",
        )
        result = classify_failure_case(evidence)
        self.assertEqual(result["primary_failure"], "citation_failure")

    def test_07_execution_failure_is_separate_from_quality(self):
        result = classify_failure_case(
            answerable_evidence(
                execution_status="failed",
                execution_subtype="provider_error",
            )
        )
        self.assertEqual(result["primary_failure"], "execution_failure")
        self.assertEqual(result["failure_subtype"], "provider_error")

    def test_08_unanswerable_hallucination_is_not_retrieval_failure(self):
        result = classify_failure_case(
            {
                "execution_status": "success",
                "answerable": False,
                "unanswerable_outcome": "unsupported_answer",
                "citation_quality": "incorrect",
            }
        )
        self.assertEqual(result["primary_failure"], "generation_failure")
        self.assertEqual(result["contributing_failures"], ["citation_failure"])

    def test_09_correct_unanswerable_abstention_is_no_failure(self):
        result = classify_failure_case(
            {
                "execution_status": "success",
                "answerable": False,
                "unanswerable_outcome": "correct_abstention",
            }
        )
        self.assertEqual(result["primary_failure"], "no_failure")

    def test_10_primary_and_contributing_failures_are_distinct(self):
        result = classify_failure_case(
            answerable_evidence(citation_quality="incorrect")
        )
        self.assertEqual(result["primary_failure"], "generation_failure")
        self.assertEqual(result["contributing_failures"], ["citation_failure"])

    def test_11_earliest_causally_sufficient_failure_wins(self):
        evidence = answerable_evidence(
            citation_quality="incorrect",
            candidate_ids_by_field={
                "chunk_id": ["other"],
                "filename": ["other.md"],
            },
        )
        result = classify_failure_case(evidence)
        self.assertEqual(result["primary_failure"], "retrieval_failure")
        self.assertNotEqual(result["primary_failure"], "generation_failure")

    def test_12_missing_candidate_or_context_evidence_needs_review(self):
        missing_candidates = classify_failure_case(
            answerable_evidence(
                candidate_ids_by_field={"chunk_id": None, "filename": None}
            )
        )
        missing_context = classify_failure_case(
            answerable_evidence(
                context_ids_by_field={"chunk_id": None, "filename": None}
            )
        )
        self.assertEqual(missing_candidates["primary_failure"], "needs_review")
        self.assertEqual(missing_context["primary_failure"], "needs_review")

    def test_13_ground_truth_granularity_controls_evidence_strength(self):
        document_only = answerable_evidence(
            expected_chunk_ids=[],
            candidate_ids_by_field={
                "chunk_id": ["other"],
                "filename": ["other.md"],
            },
        )
        strict_chunk = answerable_evidence(
            candidate_ids_by_field={
                "chunk_id": ["other"],
                "filename": ["policy.md"],
            }
        )
        self.assertEqual(
            classify_failure_case(document_only)["evidence_strength"], "supported"
        )
        self.assertEqual(
            classify_failure_case(strict_chunk)["evidence_strength"], "confirmed"
        )

    def test_14_metric_false_positive_is_not_generation_failure(self):
        evidence = answerable_evidence(
            observed_signals=["required_keyword_proxy_miss"],
            answer_quality="acceptable",
            citation_quality="acceptable",
        )
        result = classify_failure_case(evidence)
        self.assertEqual(result["primary_failure"], "no_failure")
        self.assertIn("metric", result["causal_basis"].casefold())

    def test_15_building_derived_analysis_does_not_modify_sources(self):
        loaded, evaluation, unanswerable, events, identities = self._real_sources()
        source_paths = [
            loaded["evaluation_path"],
            loaded["unanswerable_path"],
            *loaded["runtime_paths"],
        ]
        before = {path: file_sha256(path) for path in source_paths}
        build_failure_analysis_artifact(
            evaluation=evaluation,
            unanswerable=unanswerable,
            manifest=loaded["manifest"],
            manifest_identity={
                "path": "evals/failure_analysis_config.json",
                "sha256": loaded["manifest_sha256"],
            },
            runtime_events=events,
            runtime_identities=identities,
            run_id="test-run",
            repository_state={"commit": "test", "worktree_dirty": True},
        )
        self.assertEqual(before, {path: file_sha256(path) for path in source_paths})

    def test_16_primary_aggregate_sums_to_analyzed_cases(self):
        cases = [
            {"primary_failure": "retrieval_failure", "contributing_failures": [], "answerable": True},
            {"primary_failure": "no_failure", "contributing_failures": [], "answerable": True},
            {"primary_failure": "needs_review", "contributing_failures": [], "answerable": True},
        ]
        aggregate = aggregate_classifications(
            cases, observed_signal_count=2, source_counts={"raw": 3}
        )
        self.assertEqual(
            sum(aggregate["primary_failure_counts"].values()), len(cases)
        )

    def test_17_contributing_counts_can_exceed_case_count(self):
        cases = [
            {
                "primary_failure": "generation_failure",
                "contributing_failures": ["citation_failure", "ranking_failure"],
                "answerable": True,
            }
        ]
        aggregate = aggregate_classifications(
            cases, observed_signal_count=1, source_counts={"raw": 1}
        )
        self.assertEqual(sum(aggregate["contributing_failure_counts"].values()), 2)

    def test_18_needs_review_is_counted_explicitly(self):
        cases = [
            {"primary_failure": "needs_review", "contributing_failures": [], "answerable": True}
        ]
        aggregate = aggregate_classifications(
            cases, observed_signal_count=1, source_counts={"raw": 1}
        )
        self.assertEqual(aggregate["needs_review_count"], 1)
        self.assertEqual(aggregate["classified_failure_count"], 0)

    def test_19_artifact_serialization_round_trip_and_report(self):
        artifact = {
            "run_id": "round-trip",
            "source_provenance": {
                "evaluation": {"run_id": "w8t1", "sha256": "a"},
                "unanswerable": {"run_id": "w8t2", "sha256": "b"},
                "dataset_identity": {"path": "data", "sha256": "c"},
                "corpus_identity": {
                    "chunk_index": {"path": "chunks", "sha256": "d"}
                },
                "runtime_logs": [],
            },
            "resolved_configuration": {
                "retrieval_mode": "hybrid_rerank",
                "final_top_k": 2,
                "llm": {
                    "provider": "ollama",
                    "model": "qwen3:8b",
                    "model_identity": {"digest": "digest"},
                },
            },
            "aggregate": aggregate_classifications(
                [], observed_signal_count=0, source_counts={
                    "raw_source_case_executions": 0,
                    "deduplicated_primary_quality_universe": 0,
                }
            ),
            "selection": {
                "success_control_query_ids": [],
                "cases_analyzed_from_existing_artifacts": 0,
                "targeted_reproduction_count": 0,
            },
            "cases": [],
            "metric_limitations": [],
            "observability_gaps": [],
            "future_hypotheses": [],
            "evidence_join": {"selected_case_request_joins": 0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            write_json_artifact(artifact, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["run_id"], "round-trip")
        self.assertIn("Failure Analysis Report", render_failure_analysis_report(artifact, artifact_path="artifact.json"))

    def test_20_real_sources_validate_without_ollama_and_cover_all_signals(self):
        loaded, evaluation, unanswerable, _, _ = self._real_sources()
        validate_source_artifacts(
            evaluation,
            unanswerable,
            manifest=loaded["manifest"],
        )
        self.assertEqual(
            observed_w8_t1_signal_ids(evaluation),
            loaded["manifest"]["selection"]["w8_t1_observed_signal_query_ids"],
        )
        self.assertEqual(observed_w8_t2_failure_ids(unanswerable), [])

    def _real_sources(self):
        loaded = load_failure_analysis_manifest(
            MANIFEST_PATH,
            project_root=PROJECT_ROOT,
        )
        evaluation = load_json_object(
            loaded["evaluation_path"], label="W8-T1 source artifact"
        )
        unanswerable = load_json_object(
            loaded["unanswerable_path"], label="W8-T2 source artifact"
        )
        events = []
        identities = []
        for record, path in zip(
            loaded["manifest"]["source_runtime_logs"], loaded["runtime_paths"]
        ):
            rows = load_jsonl_events(path)
            events.extend(rows)
            identities.append({**record, "event_count": len(rows)})
        return loaded, evaluation, unanswerable, events, identities


if __name__ == "__main__":
    unittest.main()
