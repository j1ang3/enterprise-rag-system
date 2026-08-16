import json
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from app.evaluation.dataset import EvaluationCase, load_evaluation_dataset
from app.evaluation.rag import RAGEvaluationConfig, RAGEvaluationRunner
from app.evaluation.unanswerable import (
    AbsenceBasis,
    UnanswerableCase,
    UnanswerableEvaluationRunner,
    aggregate_unanswerable_results,
    classify_answerable_control,
    classify_unanswerable_result,
    load_unanswerable_cases,
    load_unanswerable_manifest,
    render_unanswerable_report,
    verify_absence_against_corpus,
)
from app.services.search_service import RerankedHybridConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evals" / "business_policy_eval.jsonl"
CASES_PATH = PROJECT_ROOT / "evals" / "unanswerable_cases.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "evals" / "unanswerable_evaluation_config.json"


def load_rubric() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["rubric"]


def actual_result(
    answer: str,
    *,
    answer_mode: str = "llm",
    citations: list[dict] | None = None,
    model: str | None = "qwen3:8b",
) -> dict:
    return {
        "query_id": "u-test",
        "status": "success",
        "actual": {
            "answer": answer,
            "answer_mode": answer_mode,
            "model": model,
            "citations": [] if citations is None else citations,
        },
        "error": None,
    }


def classified_row(
    outcome: str,
    *,
    status: str = "success",
    answer_mode: str = "llm",
    misleading: bool = False,
    query_id: str = "u-test",
) -> dict:
    citation = None if status != "success" else {
        "misleading_citation_proxy": misleading,
    }
    return {
        "query_id": query_id,
        "status": status,
        "actual": {"answer_mode": answer_mode, "model": "qwen3:8b"},
        "behavior_evaluation": {
            "outcome": outcome,
            "citation_evaluation": citation,
        },
    }


def control_row(
    *,
    status: str = "success",
    false_abstention: bool | None = False,
    query_id: str = "q001",
) -> dict:
    return {
        "query_id": query_id,
        "status": status,
        "actual": {"answer_mode": "llm", "model": "qwen3:8b"},
        "control_evaluation": {
            "status": "evaluated" if status == "success" else "execution_failure",
            "false_abstention": false_abstention,
        },
    }


class UnanswerableCaseTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_evaluation_dataset(DATASET_PATH)

    def write_rows(self, root: Path, rows: list[dict]) -> Path:
        path = root / "cases.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def valid_row(self, **updates) -> dict:
        row = {
            "query_id": "w8t2-test",
            "question": "What is absent?",
            "answerable": False,
            "source": "w8_t2_supplement",
            "case_type": "absent_entity_fact",
            "expected_behavior": "abstain",
            "absence_basis": {
                "full_source_review": True,
                "full_chunk_snapshot_review": True,
                "lexical_terms": ["absent"],
                "expected_absent_phrases": ["absent fact"],
                "notes": "Reviewed against the finite test corpus.",
            },
        }
        row.update(updates)
        return row

    def test_loads_frozen_stable_and_supplemental_cases(self):
        cases = load_unanswerable_cases(CASES_PATH, stable_dataset=self.dataset)

        self.assertEqual(len(cases), 8)
        self.assertEqual([case.query_id for case in cases[:4]], ["q021", "q022", "q023", "q024"])
        self.assertEqual({case.source for case in cases}, {"stable_dataset", "w8_t2_supplement"})
        self.assertTrue(all(not case.to_evaluation_case().answerable for case in cases))

    def test_rejects_invalid_identity_content_and_duplicate_cases(self):
        invalid_rows = [
            [self.valid_row(query_id="")],
            [self.valid_row(query_id="bad id")],
            [self.valid_row(question="   ")],
            [self.valid_row(answerable=True)],
            [self.valid_row(absence_basis={})],
            [self.valid_row(), self.valid_row()],
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = self.write_rows(Path(directory), rows)
                with self.assertRaises(ValueError):
                    load_unanswerable_cases(path, stable_dataset=self.dataset)

    def test_stable_case_question_must_match_frozen_dataset(self):
        row = self.valid_row(
            query_id="q021",
            question="A changed question?",
            source="stable_dataset",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(Path(directory), [row])
            with self.assertRaisesRegex(ValueError, "drifted"):
                load_unanswerable_cases(path, stable_dataset=self.dataset)

    def test_real_manifest_validates_all_recorded_identities(self):
        bundle = load_unanswerable_manifest(MANIFEST_PATH, project_root=PROJECT_ROOT)

        self.assertEqual(bundle["manifest"]["task"], "W8-T2")
        self.assertEqual(bundle["chunk_index_path"].name, "week07_tuning_chunks.json")
        self.assertEqual(len(bundle["document_paths"]), 4)


class AbsenceVerificationTests(unittest.TestCase):
    def make_case(self) -> UnanswerableCase:
        return UnanswerableCase(
            query_id="u1",
            question="What is the dental plan?",
            source="w8_t2_supplement",
            case_type="absent_entity_fact",
            expected_behavior="abstain",
            absence_basis=AbsenceBasis(
                full_source_review=True,
                full_chunk_snapshot_review=True,
                lexical_terms=("dental",),
                expected_absent_phrases=("dental plan",),
                notes="Absent in fixture.",
            ),
        )

    def test_records_term_counts_but_requires_absent_phrases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.md"
            source.write_text("This policy says dental only as an isolated term.", encoding="utf-8")
            chunks = root / "chunks.json"
            chunks.write_text(json.dumps([{"content": "Unrelated policy text."}]), encoding="utf-8")

            evidence = verify_absence_against_corpus(
                [self.make_case()], document_paths=[source], chunk_index_path=chunks
            )

        self.assertEqual(evidence["u1"]["status"], "verified_absent_for_current_corpus_snapshot")
        self.assertEqual(evidence["u1"]["term_checks"][0]["source_occurrences"], 1)

    def test_rejects_case_when_expected_absent_phrase_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.md"
            source.write_text("The dental plan is documented here.", encoding="utf-8")
            chunks = root / "chunks.json"
            chunks.write_text(json.dumps([{"content": "The dental plan is documented here."}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "now present"):
                verify_absence_against_corpus(
                    [self.make_case()], document_paths=[source], chunk_index_path=chunks
                )


class UnanswerableRubricTests(unittest.TestCase):
    def setUp(self):
        self.rubric = load_rubric()

    def classify(self, answer: str, *, citations: list[dict] | None = None) -> dict:
        return classify_unanswerable_result(
            actual_result(answer, citations=citations), rubric=self.rubric
        )

    def test_classifies_correct_abstention(self):
        result = self.classify(
            "The provided context does not contain enough information to answer reliably."
        )
        self.assertEqual(result["outcome"], "correct_abstention")

    def test_classifies_contaminated_abstention(self):
        result = self.classify(
            "I cannot confirm this from the context, but it is probably 30 days."
        )
        self.assertEqual(result["outcome"], "contaminated_abstention")

    def test_classifies_unsupported_answer_and_misleading_pipeline_citation(self):
        result = self.classify(
            "The company provides 30 dental days.",
            citations=[{"chunk_id": "expense-1"}],
        )
        self.assertEqual(result["outcome"], "unsupported_answer")
        self.assertTrue(result["citation_evaluation"]["misleading_citation_proxy"])

    def test_preserves_ambiguous_response_for_review(self):
        result = self.classify("It depends.")
        self.assertEqual(result["outcome"], "needs_review")
        self.assertEqual(result["review_status"], "not_manually_reviewed")

    def test_execution_failure_is_not_a_behavioral_failure(self):
        result = classify_unanswerable_result(
            {"status": "failed", "error": {"category": "provider_error"}},
            rubric=self.rubric,
        )
        self.assertEqual(result["outcome"], "execution_failure")
        self.assertIsNone(result["citation_evaluation"])

    def test_answerable_control_detects_false_abstention(self):
        result = classify_answerable_control(
            actual_result("The knowledge base does not contain enough information."),
            rubric=self.rubric,
        )
        self.assertTrue(result["false_abstention"])

    def test_aggregate_uses_explicit_success_and_failure_denominators(self):
        rows = [
            classified_row("correct_abstention", query_id="u1"),
            classified_row("contaminated_abstention", misleading=True, query_id="u2"),
            classified_row("needs_review", answer_mode="no_context", query_id="u3"),
            classified_row("execution_failure", status="failed", query_id="u4"),
        ]
        aggregate = aggregate_unanswerable_results(
            rows,
            [control_row(), control_row(false_abstention=True, query_id="q002")],
        )

        unanswerable = aggregate["unanswerable"]
        self.assertEqual(unanswerable["successful"], 3)
        self.assertEqual(unanswerable["failed"], 1)
        self.assertAlmostEqual(unanswerable["strict_abstention_rate"], 1 / 3)
        self.assertAlmostEqual(unanswerable["unsupported_answer_rate_including_contaminated"], 1 / 3)
        self.assertEqual(aggregate["llm_unanswerable_subset"]["sample_count"], 2)
        self.assertEqual(aggregate["answerable_controls"]["false_abstention_rate"], 0.5)


class UnanswerableRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.config = RAGEvaluationConfig(
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
                "model_identity": {"digest": "frozen-digest"},
            },
            user_id=UUID("00000000-0000-0000-0000-000000000603"),
        )
        self.unanswerable_case = UnanswerableCase(
            query_id="u1",
            question="Unknown fact?",
            source="w8_t2_supplement",
            case_type="absent_entity_fact",
            expected_behavior="abstain",
            absence_basis=AbsenceBasis(True, True, ("unknown",), ("unknown fact",), "Absent."),
        )
        self.control = EvaluationCase(
            query_id="q1",
            question="Known fact?",
            expected_answer="Known answer.",
            expected_sources=("policy.md",),
            expected_source_match="any",
            expected_citation_chunk_ids=(),
            expected_keywords=("known",),
            category="test",
            difficulty="easy",
            answerable=True,
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def rag_payload(self, question: str, *, fallback: bool = False) -> dict:
        chunk = {
            "chunk_id": "policy-1",
            "document_id": "policy",
            "filename": "policy.md",
            "content": "Known policy content.",
            "score": 0.03,
            "fused_score": 0.03,
            "rerank_score": 7.0,
            "source_ranks": {"vector": 1, "bm25": 1},
            "retrieval_mode": "hybrid_rerank",
            "context_role": "retrieved",
        }
        unknown = question == "Unknown fact?"
        return {
            "answer": (
                "The provided context does not contain enough information."
                if unknown
                else "The known answer is in policy."
            ),
            "answer_mode": "local_fallback" if fallback else "llm",
            "model": "qwen3:8b",
            "llm_error": "provider failed" if fallback else None,
            "citations": [{"chunk_id": "policy-1", "filename": "policy.md"}],
            "contexts": [chunk],
            "retrieval_evidence": {
                "configuration": RerankedHybridConfig().to_dict(),
                "candidates_before_rerank": [chunk],
                "results_after_rerank": [chunk],
            },
            "retrieval_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "total_latency_ms": 3.0,
        }

    def test_runner_reuses_w8_t1_case_execution_and_serializes_artifact(self):
        calls = []

        def fake_rag(question, *args, **kwargs):
            calls.append(question)
            return self.rag_payload(question)

        runner = UnanswerableEvaluationRunner(
            RAGEvaluationRunner(self.config, rag_callable=fake_rag),
            rubric=load_rubric(),
        )
        artifact = runner.run(
            unanswerable_cases=[self.unanswerable_case],
            absence_verification={"u1": {"status": "verified"}},
            control_cases=[self.control],
            run_id="w8-t2-test",
            run_metadata={"classification": {"llm_judge": False}},
            source_identities={
                "stable_dataset": {
                    "path": "dataset.jsonl",
                    "sha256": "dataset-hash",
                    "query_count": 2,
                },
                "unanswerable_case_file": {
                    "path": "unanswerable.jsonl",
                    "sha256": "case-hash",
                    "query_ids": ["u1"],
                },
                "corpus": {
                    "documents": [{"path": "policy.md", "sha256": "doc-hash"}],
                    "indexed_chunk_count": 1,
                    "chunk_index": {
                        "path": "chunks.json",
                        "sha256": "chunk-hash",
                    },
                },
            },
            project_root=self.root,
        )
        serialized = json.loads(json.dumps(artifact))
        report = render_unanswerable_report(serialized, artifact_path="run.json")

        self.assertEqual(calls, ["Unknown fact?", "Known fact?"])
        self.assertEqual(serialized["task"], "W8-T2")
        self.assertEqual(serialized["aggregate"]["unanswerable"]["strict_abstention_rate"], 1.0)
        self.assertIn("Correct Abstention: `u1`", report)
        self.assertIn("Unsupported Answer: None observed.", report)

    def test_formal_runner_rejects_silent_provider_fallback(self):
        base = RAGEvaluationRunner(
            self.config,
            rag_callable=lambda question, *args, **kwargs: self.rag_payload(
                question, fallback=True
            ),
        )
        result = base.run_case(self.unanswerable_case.to_evaluation_case())

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["category"], "provider_error")
        classified = classify_unanswerable_result(result, rubric=load_rubric())
        self.assertEqual(classified["outcome"], "execution_failure")


if __name__ == "__main__":
    unittest.main()
