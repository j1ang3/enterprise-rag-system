import json
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.generation import (
    calculate_citation_metrics,
    calculate_required_keyword_proxy,
)
from app.evaluation.rag import (
    RAGEvaluationConfig,
    RAGEvaluationRunner,
    render_evaluation_report,
    write_artifact,
)
from app.services.search_service import RerankedHybridConfig


def case_row(
    *,
    question: str = "How many leave days?",
    should_answer: bool = True,
    query_id: str | None = None,
) -> dict:
    row = {
        "question": question,
        "expected_answer": "Employees receive 15 leave days.",
        "expected_sources": ["hr_policy.md"] if should_answer else [],
        "expected_keywords": ["15 leave days"] if should_answer else [],
        "category": "hr_policy" if should_answer else "no_answer",
        "difficulty": "easy",
        "should_answer": should_answer,
    }
    if query_id is not None:
        row["query_id"] = query_id
    return row


def rag_result(
    *,
    answer_mode: str = "llm",
    model: str | None = "qwen3:8b",
    answer: str = "Employees receive 15 leave days.",
) -> dict:
    final = {
        "chunk_id": "eval-hr-policy-1",
        "document_id": "eval-hr-policy",
        "filename": "hr_policy.md",
        "content": "Employees receive 15 leave days.",
        "score": 0.03,
        "fused_score": 0.03,
        "rerank_score": 8.1,
        "source_ranks": {"vector": 1, "bm25": 1},
        "retrieval_mode": "hybrid_rerank",
        "context_role": "retrieved",
    }
    return {
        "answer": answer,
        "answer_mode": answer_mode,
        "model": model,
        "llm_error": None if answer_mode == "llm" else "provider failed",
        "citations": [
            {
                "chunk_id": final["chunk_id"],
                "filename": final["filename"],
            }
        ],
        "contexts": [final],
        "retrieval_evidence": {
            "configuration": {
                "per_source_candidate_depth": 5,
                "rrf_k": 60,
                "rerank_candidate_count": 3,
                "final_top_k": 2,
            },
            "candidates_before_rerank": [final],
            "results_after_rerank": [final],
        },
        "retrieval_latency_ms": 2.0,
        "generation_latency_ms": 8.0,
        "total_latency_ms": 10.0,
    }


class RAGEvaluationDatasetTests(unittest.TestCase):
    def write_rows(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "dataset.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_valid_dataset_with_stable_derived_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(
                Path(directory),
                [case_row(), case_row(question="Unknown?", should_answer=False)],
            )
            first = load_evaluation_dataset(path)
            second = load_evaluation_dataset(path)

        self.assertEqual([case.query_id for case in first.cases], ["q001", "q002"])
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.identity()["schema_version"], "business_policy_eval.v1")

    def test_rejects_empty_question_duplicate_id_and_malformed_labels(self):
        invalid_rows = [
            [case_row(question="   ")],
            [case_row(query_id="same"), case_row(query_id="same")],
            [{**case_row(), "expected_sources": "hr_policy.md"}],
            [{**case_row(), "unknown": True}],
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = self.write_rows(Path(directory), rows)
                with self.assertRaises(ValueError):
                    load_evaluation_dataset(path)

    def test_rejects_missing_file_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_evaluation_dataset(root / "missing.jsonl")
            broken = root / "broken.jsonl"
            broken.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                load_evaluation_dataset(broken)


class GenerationMetricTests(unittest.TestCase):
    def test_required_keyword_proxy_is_explicitly_lexical(self):
        correct = calculate_required_keyword_proxy(
            "Employees receive 15 LEAVE days.", ["15 leave days"]
        )
        variant = calculate_required_keyword_proxy(
            "Employees receive fifteen leave days.", ["15 leave days"]
        )

        self.assertTrue(correct["matched"])
        self.assertFalse(variant["matched"])
        self.assertEqual(variant["recall"], 0.0)

    def test_document_citation_metrics_handle_extra_and_missing_sources(self):
        exact = calculate_citation_metrics(
            [{"filename": "a.md", "chunk_id": "a1"}],
            expected_documents=["a.md"],
            expected_chunk_ids=["a1"],
        )
        extra = calculate_citation_metrics(
            [
                {"filename": "a.md", "chunk_id": "a1"},
                {"filename": "b.md", "chunk_id": "b1"},
            ],
            expected_documents=["a.md"],
            expected_chunk_ids=["a1"],
        )

        self.assertTrue(exact["document"]["exact_match"])
        self.assertEqual(exact["strict_chunk"]["recall"], 1.0)
        self.assertFalse(extra["document"]["exact_match"])
        self.assertEqual(extra["document"]["precision"], 0.5)


class RAGEvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.dataset_path = self.root / "dataset.jsonl"
        self.dataset_path.write_text(
            json.dumps(case_row()) + "\n",
            encoding="utf-8",
        )
        self.dataset = load_evaluation_dataset(self.dataset_path)
        self.configuration = RAGEvaluationConfig(
            formal=True,
            retrieval_mode="hybrid_rerank",
            top_k=2,
            metric_k_values=(1, 2),
            index_path=self.root / "chunks.json",
            vector_index_path=self.root / "vectors.json",
            reranked_hybrid=RerankedHybridConfig(),
            llm_metadata={
                "provider": "ollama",
                "model": "qwen3:8b",
                "temperature": 0.2,
                "max_tokens": 512,
                "model_identity": {"digest": "digest"},
            },
            user_id=UUID("00000000-0000-0000-0000-000000000601"),
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_invokes_production_rag_contract_and_records_per_query_evidence(self):
        calls = []

        def fake_rag(question, top_k, **kwargs):
            calls.append((question, top_k, kwargs))
            return rag_result()

        artifact = RAGEvaluationRunner(
            self.configuration, rag_callable=fake_rag
        ).run(
            self.dataset,
            run_id="run-1",
            run_metadata={"repository": {"dirty": True}},
            project_root=self.root,
        )

        self.assertEqual(calls[0][2]["retrieval_mode"], "hybrid_rerank")
        self.assertIs(calls[0][2]["reranked_hybrid_config"], self.configuration.reranked_hybrid)
        result = artifact["results"][0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actual"]["retrieval"]["final_chunks"][0]["chunk_id"], "eval-hr-policy-1")
        self.assertEqual(result["metrics"]["retrieval"]["metrics_by_k"]["1"]["recall"], 1.0)
        self.assertEqual(artifact["aggregate"]["case_counts"]["successful"], 1)

    def test_formal_model_identity_rejects_gemma(self):
        with self.assertRaisesRegex(ValueError, "qwen3:8b"):
            RAGEvaluationConfig(
                formal=True,
                retrieval_mode="hybrid_rerank",
                top_k=2,
                metric_k_values=(1, 2),
                index_path=self.root / "chunks.json",
                vector_index_path=self.root / "vectors.json",
                reranked_hybrid=RerankedHybridConfig(),
                llm_metadata={
                    "provider": "ollama",
                    "model": "gemma3:4b",
                    "model_identity": {"digest": "digest"},
                },
                user_id=UUID("00000000-0000-0000-0000-000000000601"),
            )

    def test_qwen_fallback_is_failed_without_silent_model_switch(self):
        fallback = rag_result(answer_mode="local_fallback")
        result = RAGEvaluationRunner(
            self.configuration, rag_callable=lambda *args, **kwargs: fallback
        ).run_case(self.dataset.cases[0])

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["category"], "provider_error")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["actual"]["model"], "qwen3:8b")

    def test_failed_execution_is_not_counted_as_zero_quality(self):
        def fail(*args, **kwargs):
            raise RuntimeError("service unavailable")

        artifact = RAGEvaluationRunner(
            self.configuration, rag_callable=fail
        ).run(
            self.dataset,
            run_id="failed-run",
            run_metadata={},
            project_root=self.root,
        )

        aggregate = artifact["aggregate"]
        self.assertEqual(aggregate["case_counts"]["failed"], 1)
        self.assertEqual(aggregate["retrieval"]["metrics_by_k"]["1"]["sample_count"], 0)
        self.assertIsNone(aggregate["generation"]["required_keyword_proxy"]["match_rate"])

    def test_unanswerable_is_preserved_but_excluded_from_generation_aggregate(self):
        self.dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(case_row()),
                    json.dumps(case_row(question="Unknown?", should_answer=False)),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = load_evaluation_dataset(self.dataset_path)

        def fake_rag(question, *args, **kwargs):
            if question == "Unknown?":
                result = rag_result(
                    answer_mode="no_context",
                    model=None,
                    answer="No reliable answer.",
                )
                result["contexts"] = []
                result["citations"] = []
                result["retrieval_evidence"]["candidates_before_rerank"] = []
                result["retrieval_evidence"]["results_after_rerank"] = []
                return result
            return rag_result()

        artifact = RAGEvaluationRunner(
            self.configuration, rag_callable=fake_rag
        ).run(
            dataset,
            run_id="mixed-run",
            run_metadata={},
            project_root=self.root,
        )

        self.assertEqual(artifact["aggregate"]["unanswerable"]["raw_output_count"], 1)
        self.assertEqual(artifact["aggregate"]["generation"]["required_keyword_proxy"]["sample_count"], 1)
        self.assertEqual(artifact["results"][1]["metrics"]["generation"]["status"], "deferred_to_W8_T2")

    def test_artifact_serialization_and_report(self):
        artifact = RAGEvaluationRunner(
            self.configuration, rag_callable=lambda *args, **kwargs: rag_result()
        ).run(
            self.dataset,
            run_id="serial-run",
            run_metadata={},
            project_root=self.root,
        )
        path = self.root / "runs" / "serial-run.json"
        write_artifact(artifact, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        report = render_evaluation_report(
            loaded, artifact_path="runs/serial-run.json"
        )

        self.assertEqual(loaded["run_id"], artifact["run_id"])
        self.assertEqual(loaded["aggregate"], artifact["aggregate"])
        self.assertIn("qwen3:8b", report)
        self.assertIn("Groundedness: not formally automated", report)
        with self.assertRaises(FileExistsError):
            write_artifact(artifact, path)


if __name__ == "__main__":
    unittest.main()
