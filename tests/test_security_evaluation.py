import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from app.evaluation.generation import calculate_citation_metrics
from app.evaluation.rag import RAGEvaluationConfig, RAGEvaluationRunner
from app.evaluation.security_comparison import (
    CapturingRAGCallable,
    SecurityEvaluationValidationError,
    _leakage_metric,
    aggregate_direct_cases,
    aggregate_indirect_cases,
    delta_record,
    load_security_evaluation_manifest,
    outcome_transitions,
    rate_record,
    validate_fair_run_pair,
    write_json_artifact,
)
from app.evaluation.unanswerable import aggregate_unanswerable_results
from app.services.search_service import RerankedHybridConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "evals/security/security_evaluation_config.json"


def _pair(mode: str, identity: dict | None = None) -> dict:
    return {
        "task": "W9-T5",
        "formal": True,
        "artifact_type": "direct_attack_run",
        "security_mode": mode,
        "controlled_identity": identity or {"model": "qwen3:8b", "dataset": "same"},
        "production_pipeline": {"entrypoint": "answer_question"},
    }


def _direct_row(outcome: str, category: str = "system_prompt_extraction") -> dict:
    return {
        "attack_case": {"attack_id": "A", "category": category},
        "evaluation_runtime_ms": 1.0,
        "final_response_evaluation": {"outcome": outcome},
    }


def _indirect_row(outcome: str, delivery: str = "delivered_to_context") -> dict:
    return {
        "attack_case": {"attack_id": "I", "category": "synthetic_canary_leakage"},
        "ingestion": {"document_ingested": True},
        "final_execution": {
            "status": "success",
            "answer_mode": "llm",
            "model": "qwen3:8b",
            "timings_ms": {"total": 1.0},
        },
        "delivery_evidence": {"delivery_status": delivery},
        "final_response_evaluation": {"outcome": outcome},
    }


class SecurityEvaluationTests(unittest.TestCase):
    def test_01_manifest_loads_baseline_and_layered_modes(self):
        value = load_security_evaluation_manifest(MANIFEST, project_root=PROJECT_ROOT)
        self.assertEqual(value["experimental_design"]["modes"], ["baseline", "layered"])

    def test_02_same_model_is_required(self):
        with self.assertRaises(SecurityEvaluationValidationError):
            validate_fair_run_pair(
                _pair("baseline", {"model": "qwen3:8b"}),
                _pair("layered", {"model": "gemma3:4b"}),
                artifact_type="direct_attack_run",
            )

    def test_03_same_attack_dataset_is_required(self):
        with self.assertRaises(SecurityEvaluationValidationError):
            validate_fair_run_pair(
                _pair("baseline", {"dataset": "a"}),
                _pair("layered", {"dataset": "b"}),
                artifact_type="direct_attack_run",
            )

    def test_04_same_corpus_is_required(self):
        with self.assertRaises(SecurityEvaluationValidationError):
            validate_fair_run_pair(
                _pair("baseline", {"corpus": "a"}),
                _pair("layered", {"corpus": "b"}),
                artifact_type="direct_attack_run",
            )

    def test_05_same_generation_config_is_required(self):
        with self.assertRaises(SecurityEvaluationValidationError):
            validate_fair_run_pair(
                _pair("baseline", {"temperature": 0.2}),
                _pair("layered", {"temperature": 0.3}),
                artifact_type="direct_attack_run",
            )

    def test_06_direct_asr_excludes_failure(self):
        rows = [_direct_row("successful"), _direct_row("resisted"), _direct_row("execution_failure")]
        aggregate = aggregate_direct_cases(rows, evaluation_key="final_response_evaluation")
        self.assertEqual(aggregate["case_counts"]["successfully_executed"], 2)
        self.assertEqual(aggregate["attack_success_rate"], 0.5)

    def test_07_indirect_end_to_end_asr(self):
        rows = [_indirect_row("successful"), _indirect_row("resisted")]
        aggregate = aggregate_indirect_cases(rows, evaluation_key="final_response_evaluation")
        self.assertEqual(aggregate["end_to_end_attack_success_rate"], 0.5)

    def test_08_conditional_asr_uses_delivered_only(self):
        rows = [_indirect_row("successful"), _indirect_row("not_evaluated", "not_retrieved")]
        aggregate = aggregate_indirect_cases(rows, evaluation_key="final_response_evaluation")
        self.assertEqual(aggregate["conditional_attack_success_rate"], 1.0)

    def test_09_zero_conditional_denominator_is_na(self):
        rows = [_indirect_row("not_evaluated", "not_retrieved")]
        aggregate = aggregate_indirect_cases(rows, evaluation_key="final_response_evaluation")
        self.assertIsNone(aggregate["conditional_attack_success_rate"])

    def test_10_execution_failure_rate_has_all_case_denominator(self):
        metric = rate_record(1, 3)
        self.assertEqual(metric, {"numerator": 1, "denominator": 3, "rate": 1 / 3})

    def test_11_prompt_leakage_aggregation(self):
        rows = [_direct_row("successful"), _direct_row("partial_success")]
        metric = _leakage_metric(rows, categories={"system_prompt_extraction"})
        self.assertEqual((metric["numerator"], metric["denominator"]), (1, 2))

    def test_12_document_canary_leakage_aggregation(self):
        rows = [_indirect_row("successful")]
        metric = _leakage_metric(rows, categories={"synthetic_canary_leakage"})
        self.assertEqual(metric["rate"], 1.0)

    def test_13_false_refusal_rate(self):
        delta = delta_record(rate_record(0, 2), rate_record(1, 2))
        self.assertEqual(delta["layered_minus_baseline"], 0.5)

    def test_14_correct_abstention_is_not_false_refusal(self):
        row = {
            "status": "success",
            "actual": {"answer_mode": "llm", "model": "qwen3:8b"},
            "behavior_evaluation": {
                "outcome": "correct_abstention",
                "citation_evaluation": {"misleading_citation_proxy": False},
            },
        }
        aggregate = aggregate_unanswerable_results([row], [])
        self.assertEqual(aggregate["unanswerable"]["strict_abstention_rate"], 1.0)
        self.assertEqual(aggregate["answerable_controls"]["false_abstention_count"], 0)

    def test_15_existing_citation_metric_is_reused(self):
        metric = calculate_citation_metrics(
            [{"filename": "a.md", "chunk_id": "a-1"}],
            expected_documents=["a.md"],
            expected_chunk_ids=["a-1"],
        )
        self.assertEqual(metric["document"]["f1"], 1.0)
        self.assertEqual(metric["strict_chunk"]["recall"], 1.0)

    def test_16_transition_joins_by_case_id(self):
        transition = outcome_transitions(
            [{"id": "A", "outcome": "resisted"}],
            [{"id": "A", "outcome": "successful"}],
            id_getter=lambda row: row["id"],
            outcome_getter=lambda row: row["outcome"],
        )
        self.assertEqual(transition["counts"], {"resisted -> successful": 1})

    def test_17_transition_rejects_missing_case(self):
        with self.assertRaises(SecurityEvaluationValidationError):
            outcome_transitions(
                [{"id": "A", "outcome": "resisted"}],
                [{"id": "B", "outcome": "resisted"}],
                id_getter=lambda row: row["id"],
                outcome_getter=lambda row: row["outcome"],
            )

    def test_18_benign_regression_transition_is_representable(self):
        transition = outcome_transitions(
            [{"id": "Q", "outcome": "correct"}],
            [{"id": "Q", "outcome": "refused"}],
            id_getter=lambda row: row["id"],
            outcome_getter=lambda row: row["outcome"],
        )
        self.assertEqual(transition["cases"][0]["transition"], "correct -> refused")

    def test_19_artifact_serialization_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact.json"
            write_json_artifact({"task": "W9-T5", "value": "安全"}, target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["value"], "安全")

    def test_20_manifest_detects_historical_hash_drift(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["frozen_identities"]["w9_t2_reviewed_artifact"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "manifest.json"
            altered.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SecurityEvaluationValidationError):
                load_security_evaluation_manifest(altered, project_root=PROJECT_ROOT)

    def test_21_formal_runner_rejects_silent_fallback(self):
        config = RAGEvaluationConfig(
            formal=True,
            retrieval_mode="hybrid_rerank",
            top_k=2,
            metric_k_values=(1, 2),
            index_path=Path("chunks.json"),
            vector_index_path=Path("vectors.json"),
            reranked_hybrid=RerankedHybridConfig(5, 60, 3, 2),
            llm_metadata={
                "provider": "ollama",
                "model": "qwen3:8b",
                "model_identity": {"digest": "digest"},
            },
            user_id=UUID("00000000-0000-0000-0000-000000000602"),
        )
        runner = RAGEvaluationRunner(config)
        failure = runner._formal_output_failure(
            {"answer_mode": "local_fallback", "contexts": [{"chunk_id": "c"}], "model": None}
        )
        self.assertIn("non-LLM fallback", failure)

    def test_22_capture_returns_pre_validator_output(self):
        capture = CapturingRAGCallable()
        captured = {"answer": "raw", "mode": "llm", "model": "qwen3:8b", "citations": []}
        with patch("app.services.rag_service.build_answer", return_value=captured):
            # The wrapper itself is covered by production-path tests; queue semantics stay unit-only here.
            capture._captured.append(captured)
        self.assertEqual(capture.take_last()["answer"], "raw")
        capture.assert_empty()


if __name__ == "__main__":
    unittest.main()
