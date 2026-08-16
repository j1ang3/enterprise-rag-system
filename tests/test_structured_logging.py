import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID
from types import SimpleNamespace

from app.core.config import settings
from app.evaluation.dataset import EvaluationCase
from app.evaluation.rag import RAGEvaluationConfig, RAGEvaluationRunner
from app.observability.rag_logging import (
    build_rag_event,
    emit_rag_event,
    new_request_id,
    serialize_rag_event,
)
from app.routers.chat import chat
from app.schemas.chat import ChatRequest
from app.services.rag_service import answer_question
from app.services.search_service import (
    RerankedHybridConfig,
    RerankedHybridSearchResult,
)


CONTEXT = {
    "chunk_id": "chunk-1",
    "document_id": "document-1",
    "filename": "policy.md",
    "position": 1,
    "content": "CONTEXT_SECRET employees receive 15 leave days.",
    "score": 0.7,
    "fused_score": 0.03,
    "rerank_score": 8.2,
    "source_ranks": {"vector": 1, "bm25": 2},
    "retrieval_mode": "hybrid_rerank",
    "context_role": "retrieved",
}
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000502")


def answer_result(*, usage=True, error=False):
    return {
        "answer": "ANSWER_SECRET employees receive 15 days.",
        "mode": "local_fallback" if error else "llm",
        "model": "qwen3:8b",
        "llm_error": "LLM provider is temporarily unavailable." if error else None,
        "llm_error_code": "provider_unavailable" if error else None,
        "llm_latency_ms": 7.5,
        "llm_usage": (
            {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
            if usage
            else None
        ),
        "citations": [{"chunk_id": "chunk-1", "document_id": "document-1"}],
    }


class StructuredLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temporary_directory.name) / "rag.jsonl"
        self.settings_patches = (
            patch.object(settings, "rag_structured_logging_enabled", True),
            patch.object(settings, "rag_structured_log_path", self.log_path),
            patch.object(settings, "llm_provider", "ollama"),
            patch(
                "app.services.rag_service.get_readable_document_ids",
                return_value=frozenset({"document-1", "document-2"}),
            ),
        )
        for setting_patch in self.settings_patches:
            setting_patch.start()

    def tearDown(self):
        for setting_patch in reversed(self.settings_patches):
            setting_patch.stop()
        self.temporary_directory.cleanup()

    def read_events(self):
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
        ]

    def run_service(self, *, result=None, question="QUERY_SECRET leave days?"):
        with patch(
            "app.services.rag_service.search_chunks", return_value=[CONTEXT]
        ), patch(
            "app.services.rag_service.build_answer",
            return_value=result or answer_result(),
        ):
            return answer_question(
                question,
                retrieval_mode="keyword",
                user_id=TEST_USER_ID,
            )

    def test_request_ids_are_valid_and_unique(self):
        first = new_request_id()
        second = new_request_id()
        self.assertNotEqual(first, second)
        self.assertEqual(str(UUID(first)), first)
        self.assertEqual(str(UUID(second)), second)

    def test_router_propagates_one_request_id(self):
        request_id = "9a058232-f3fe-45db-b0bf-eb622d7f70d0"
        service_result = {
            "question": "Policy?",
            "retrieval_mode": "keyword",
            "min_score": 0.2,
            "answer": "Answer",
            "answer_mode": "no_context",
            "model": None,
            "llm_error": None,
            "citations": [],
            "contexts": [],
        }
        with patch("app.routers.chat.new_request_id", return_value=request_id), patch(
            "app.routers.chat.answer_question", return_value=service_result
        ) as rag:
            response = chat(
                ChatRequest(question="Policy?"),
                SimpleNamespace(user_id=TEST_USER_ID),
            )

        self.assertEqual(response["data"]["request_id"], request_id)
        self.assertEqual(rag.call_args.kwargs["request_id"], request_id)

    def test_success_event_has_identity_evidence_timing_and_usage(self):
        result = self.run_service()
        event = self.read_events()[0]

        self.assertEqual(event, result["runtime_event"])
        self.assertEqual(event["request_id"], result["request_id"])
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["provider"], "ollama")
        self.assertEqual(event["model"], "qwen3:8b")
        self.assertEqual(event["retrieved_chunk_ids"], ["chunk-1"])
        self.assertEqual(event["retrieved_document_ids"], ["document-1"])
        self.assertEqual(event["cited_chunk_ids"], ["chunk-1"])
        self.assertEqual(event["total_tokens"], 25)
        self.assertGreaterEqual(event["retrieval_ms"], 0)
        self.assertGreaterEqual(event["generation_ms"], 0)
        self.assertGreaterEqual(event["total_ms"], event["retrieval_ms"])

    def test_content_and_secrets_are_not_logged(self):
        self.run_service()
        serialized = self.log_path.read_text(encoding="utf-8")
        self.assertNotIn("QUERY_SECRET", serialized)
        self.assertNotIn("CONTEXT_SECRET", serialized)
        self.assertNotIn("ANSWER_SECRET", serialized)
        self.assertNotIn("api_key", serialized.casefold())
        self.assertNotIn("content", serialized.casefold())

    def test_provider_failure_is_failed_llm_event_with_safe_message(self):
        result = self.run_service(result=answer_result(error=True))
        event = self.read_events()[0]
        self.assertEqual(result["answer_mode"], "local_fallback")
        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["error_stage"], "llm")
        self.assertEqual(event["error_type"], "provider_unavailable")
        self.assertEqual(
            event["error_message"], "LLM provider is temporarily unavailable."
        )

    def test_missing_usage_is_explicit_null(self):
        self.run_service(result=answer_result(usage=False))
        event = self.read_events()[0]
        self.assertIsNone(event["prompt_tokens"])
        self.assertIsNone(event["completion_tokens"])
        self.assertIsNone(event["total_tokens"])

    def test_hybrid_rerank_uses_component_timing_boundaries(self):
        configuration = RerankedHybridConfig()
        trace = RerankedHybridSearchResult(
            candidates_before_rerank=[CONTEXT],
            results_after_rerank=[CONTEXT],
            configuration=configuration,
            retrieval_ms=12.25,
            rerank_ms=4.5,
        )
        with patch(
            "app.services.rag_service.run_reranked_hybrid_search", return_value=trace
        ), patch(
            "app.services.rag_service.finalize_contexts", return_value=[CONTEXT]
        ), patch(
            "app.services.rag_service.build_answer", return_value=answer_result()
        ):
            answer_question(
                "How many leave days?",
                2,
                retrieval_mode="hybrid_rerank",
                reranked_hybrid_config=configuration,
                user_id=TEST_USER_ID,
            )

        event = self.read_events()[0]
        self.assertEqual(event["retrieval_ms"], 12.25)
        self.assertEqual(event["rerank_ms"], 4.5)
        self.assertGreaterEqual(event["context_build_ms"], 0)

    def test_unhandled_exception_is_logged_then_re_raised_without_details(self):
        with patch(
            "app.services.rag_service.search_chunks",
            side_effect=RuntimeError("SECRET_INTERNAL_DETAIL"),
        ):
            with self.assertRaisesRegex(RuntimeError, "SECRET_INTERNAL_DETAIL"):
                answer_question(
                    "Question?",
                    retrieval_mode="keyword",
                    user_id=TEST_USER_ID,
                )

        event = self.read_events()[0]
        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["error_stage"], "retrieval")
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertNotIn("SECRET_INTERNAL_DETAIL", json.dumps(event))

    def test_logger_write_failure_does_not_change_business_result(self):
        event = build_rag_event(
            request_id=new_request_id(),
            status="success",
            retrieval_mode="keyword",
            provider="ollama",
            model="qwen3:8b",
            answer_mode="llm",
            query_length=3,
            retrieved_results=[],
            contexts=[],
            citations=[],
            retrieval_ms=1,
            rerank_ms=None,
            context_build_ms=None,
            generation_ms=2,
            llm_ms=1.5,
            total_ms=3,
            token_usage=None,
        )
        with patch(
            "app.observability.rag_logging._write_json_line",
            side_effect=OSError("disk unavailable"),
        ):
            self.assertFalse(emit_rag_event(event))

        with patch(
            "app.services.rag_service.build_rag_event",
            side_effect=ValueError("serialization failed"),
        ):
            result = self.run_service()
        self.assertEqual(result["answer"], "ANSWER_SECRET employees receive 15 days.")
        self.assertIsNone(result["runtime_event"])

    def test_event_is_one_line_valid_json(self):
        result = self.run_service()
        payload = serialize_rag_event(result["runtime_event"])
        self.assertNotIn("\n", payload)
        self.assertEqual(json.loads(payload)["schema_version"], 2)

    def test_logging_does_not_mutate_answer_or_retrieval_order(self):
        second = {**CONTEXT, "chunk_id": "chunk-2", "document_id": "document-2"}
        expected_contexts = [CONTEXT, second]
        with patch(
            "app.services.rag_service.search_chunks", return_value=expected_contexts
        ), patch(
            "app.services.rag_service.build_answer", return_value=answer_result()
        ):
            enabled = answer_question(
                "Question?",
                retrieval_mode="keyword",
                user_id=TEST_USER_ID,
            )
            with patch.object(settings, "rag_structured_logging_enabled", False):
                disabled = answer_question(
                    "Question?",
                    retrieval_mode="keyword",
                    user_id=TEST_USER_ID,
                )

        self.assertEqual(enabled["answer"], disabled["answer"])
        self.assertEqual(
            [item["chunk_id"] for item in enabled["contexts"]],
            [item["chunk_id"] for item in disabled["contexts"]],
        )
        self.assertEqual(enabled["contexts"], expected_contexts)

    def test_evaluation_runner_uses_same_production_logging_path(self):
        configuration = RerankedHybridConfig()
        trace = RerankedHybridSearchResult(
            candidates_before_rerank=[CONTEXT],
            results_after_rerank=[CONTEXT],
            configuration=configuration,
            retrieval_ms=1.0,
            rerank_ms=2.0,
        )
        evaluation_config = RAGEvaluationConfig(
            formal=False,
            retrieval_mode="hybrid_rerank",
            top_k=2,
            metric_k_values=(1, 2),
            index_path=Path("unused-chunks.json"),
            vector_index_path=Path("unused-vectors.json"),
            reranked_hybrid=configuration,
            llm_metadata={"provider": "ollama", "model": "qwen3:8b"},
            user_id=TEST_USER_ID,
        )
        case = EvaluationCase(
            query_id="q001",
            question="How many leave days?",
            expected_answer="15 days",
            expected_sources=("policy.md",),
            expected_source_match="filename",
            expected_citation_chunk_ids=(),
            expected_keywords=("15 days",),
            category="policy",
            difficulty="easy",
            answerable=True,
        )
        with patch(
            "app.services.rag_service.run_reranked_hybrid_search", return_value=trace
        ), patch(
            "app.services.rag_service.finalize_contexts", return_value=[CONTEXT]
        ), patch(
            "app.services.rag_service.build_answer", return_value=answer_result()
        ):
            evaluation = RAGEvaluationRunner(
                evaluation_config, rag_callable=answer_question
            ).run_case(case)

        self.assertEqual(evaluation["status"], "success")
        self.assertEqual(
            evaluation["actual"]["request_id"], self.read_events()[0]["request_id"]
        )
        self.assertEqual(evaluation["actual"]["runtime_event"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
