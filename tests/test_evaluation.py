import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from app.core.config import settings
from scripts.evaluate_rag import (
    disable_llm_for_evaluation,
    evaluate_example,
    load_dataset,
    summarize,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DATASET = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000604")


class EvaluationTests(unittest.TestCase):
    def test_disable_llm_overrides_loaded_env_credentials(self):
        previous_key = settings.llm_api_key
        settings.llm_api_key = "loaded-from-dotenv"
        try:
            disable_llm_for_evaluation()
            self.assertEqual(settings.llm_api_key, "")
        finally:
            settings.llm_api_key = previous_key

    def test_baseline_dataset_has_required_size_and_difficulty_mix(self):
        examples = load_dataset(BASELINE_DATASET)

        self.assertGreaterEqual(len(examples), 20)
        self.assertLessEqual(len(examples), 30)
        self.assertEqual(
            {example["difficulty"] for example in examples},
            {"easy", "medium", "hard"},
        )
        self.assertTrue(any(not example["should_answer"] for example in examples))
        self.assertTrue(
            any(
                example["difficulty"] == "hard" and example["should_answer"]
                for example in examples
            )
        )

    def test_baseline_dataset_has_traceable_expected_fields(self):
        examples = load_dataset(BASELINE_DATASET)
        required_fields = {
            "question",
            "expected_answer",
            "expected_sources",
            "expected_keywords",
            "category",
            "difficulty",
            "should_answer",
        }

        for example in examples:
            with self.subTest(question=example.get("question")):
                self.assertTrue(required_fields.issubset(example))
                self.assertTrue(example["question"].strip())
                self.assertTrue(example["expected_answer"].strip())

    def test_evaluate_example_uses_rag_service_and_records_latency(self):
        rag_result = {
            "answer": "Employees receive 15 days.",
            "answer_mode": "llm",
            "model": "qwen3:8b",
            "llm_error": None,
            "min_score": 0.2,
            "citations": [
                {
                    "chunk_id": "eval-hr-policy-1",
                    "filename": "hr_policy.md",
                }
            ],
            "contexts": [
                {
                    "chunk_id": "eval-hr-policy-1",
                    "filename": "hr_policy.md",
                    "score": 0.9,
                }
            ],
            "retrieval_latency_ms": 2.5,
            "generation_latency_ms": 10.0,
            "total_latency_ms": 12.7,
        }
        example = {
            "question": "How many leave days?",
            "expected_sources": ["hr_policy.md"],
            "expected_citation_chunk_ids": ["eval-hr-policy-1"],
            "expected_keywords": ["15 days"],
            "should_answer": True,
        }

        with patch("scripts.evaluate_rag.answer_question", return_value=rag_result) as rag:
            result = evaluate_example(
                example,
                3,
                None,
                None,
                "vector",
                0.2,
                user_id=TEST_USER_ID,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["retrieval_latency_ms"], 2.5)
        self.assertEqual(result["min_score"], 0.2)
        self.assertEqual(result["generation_latency_ms"], 10.0)
        self.assertEqual(result["total_latency_ms"], 12.7)
        self.assertEqual(result["model"], "qwen3:8b")
        rag.assert_called_once()

    def test_require_llm_rejects_local_fallback_with_context(self):
        rag_result = {
            "answer": "Fallback context",
            "answer_mode": "local_fallback",
            "model": "qwen3:8b",
            "llm_error": "LLM provider request timed out.",
            "min_score": 0.2,
            "citations": [],
            "contexts": [{"chunk_id": "chunk-1", "filename": "policy.md"}],
            "retrieval_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "total_latency_ms": 3.0,
        }
        example = {
            "question": "What is the policy?",
            "expected_sources": ["policy.md"],
            "expected_keywords": [],
            "should_answer": True,
        }

        with patch("scripts.evaluate_rag.answer_question", return_value=rag_result):
            with self.assertRaisesRegex(RuntimeError, "Formal LLM evaluation stopped"):
                evaluate_example(
                    example,
                    3,
                    None,
                    None,
                    "vector",
                    0.2,
                    require_llm=True,
                    user_id=TEST_USER_ID,
                )

    def test_summary_reports_quality_latency_and_answer_modes(self):
        base = {
            "should_answer": True,
            "passed": True,
            "source_hit": True,
            "keyword_hit": True,
            "no_answer_hit": True,
            "citation_hit": True,
            "citation_source_hit": True,
            "citation_chunk_hit": True,
            "answer_mode": "llm",
            "model": "qwen3:8b",
            "retrieval_latency_ms": 2.0,
            "generation_latency_ms": 8.0,
            "total_latency_ms": 10.0,
        }
        unanswerable = {
            **base,
            "should_answer": False,
            "answer_mode": "no_context",
            "retrieval_latency_ms": 4.0,
            "generation_latency_ms": 0.0,
            "total_latency_ms": 4.0,
        }

        summary = summarize([base, unanswerable])

        self.assertEqual(summary["retrieval_recall_at_k"], 1.0)
        self.assertEqual(summary["answer_correctness_proxy"], 1.0)
        self.assertEqual(summary["citation_source_accuracy"], 1.0)
        self.assertEqual(summary["average_retrieval_latency_ms"], 3.0)
        self.assertEqual(summary["average_generation_latency_ms"], 4.0)
        self.assertEqual(summary["average_total_latency_ms"], 7.0)
        self.assertEqual(summary["answer_mode_counts"], {"llm": 1, "no_context": 1})
        self.assertEqual(summary["model_counts"], {"qwen3:8b": 2})


if __name__ == "__main__":
    unittest.main()
