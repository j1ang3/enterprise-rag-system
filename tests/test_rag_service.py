import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from app.services.access_control import AccessControlUnavailableError
from app.services.knowledge_base import build_answer, build_citations
from app.services.rag_service import answer_question, resolve_min_score
from app.services.search_service import (
    RerankedHybridConfig,
    RerankedHybridSearchResult,
)


CONTEXT = {
    "chunk_id": "policy-1",
    "document_id": "policy",
    "filename": "policy.pdf",
    "position": 1,
    "chunk_index": 0,
    "page_number": 7,
    "content": "Employees receive 15 paid annual leave days.",
    "score": 0.91,
    "retrieval_mode": "vector",
    "context_role": "retrieved",
}
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000301")


class RagServiceTests(unittest.TestCase):
    def test_answer_question_connects_retrieval_and_generation(self):
        answer = {
            "answer": "Employees receive 15 days.",
            "mode": "llm",
            "model": "test-model",
            "llm_error": None,
            "citations": [{"chunk_id": "policy-1"}],
        }

        with patch(
            "app.services.rag_service.search_chunks",
            return_value=[CONTEXT],
        ) as search, patch(
            "app.services.rag_service.build_answer",
            return_value=answer,
        ) as generate, patch(
            "app.services.rag_service.get_readable_document_ids",
            return_value=frozenset({"policy"}),
        ):
            result = answer_question(
                "How many leave days?",
                2,
                retrieval_mode="vector",
                min_score=0.3,
                user_id=TEST_USER_ID,
            )

        search.assert_called_once_with(
            "How many leave days?",
            2,
            index_path=None,
            retrieval_mode="vector",
            vector_index_path=None,
            min_score=0.3,
            allowed_document_ids=frozenset({"policy"}),
        )
        generate.assert_called_once_with(
            "How many leave days?",
            [CONTEXT],
            security_mode="layered",
        )
        self.assertEqual(result["answer_mode"], "llm")
        self.assertEqual(result["contexts"], [CONTEXT])
        self.assertGreaterEqual(result["retrieval_latency_ms"], 0.0)
        self.assertGreaterEqual(result["generation_latency_ms"], 0.0)
        self.assertGreaterEqual(result["total_latency_ms"], 0.0)

    def test_no_context_never_calls_llm_provider(self):
        with patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch("app.services.knowledge_base.chat_completion") as completion:
            result = build_answer("Unknown question", [])

        self.assertEqual(result["mode"], "no_context")
        self.assertIn("could not find relevant content", result["answer"])
        self.assertEqual(result["citations"], [])
        completion.assert_not_called()

    def test_zero_access_is_deny_all_and_never_calls_llm(self):
        with patch(
            "app.services.rag_service.get_readable_document_ids",
            return_value=frozenset(),
        ), patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch(
            "app.services.knowledge_base.chat_completion",
        ) as completion:
            result = answer_question(
                "annual leave",
                3,
                retrieval_mode="keyword",
                user_id=TEST_USER_ID,
            )

        self.assertEqual(result["contexts"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["answer_mode"], "no_context")
        completion.assert_not_called()

    def test_authorization_database_failure_stops_before_retrieval(self):
        with patch(
            "app.services.rag_service.get_readable_document_ids",
            side_effect=AccessControlUnavailableError("private database detail"),
        ), patch("app.services.rag_service.search_chunks") as search, patch(
            "app.services.rag_service.build_answer"
        ) as generate:
            with self.assertRaises(AccessControlUnavailableError):
                answer_question(
                    "annual leave",
                    3,
                    retrieval_mode="keyword",
                    user_id=TEST_USER_ID,
                )

        search.assert_not_called()
        generate.assert_not_called()

    def test_unauthorized_text_cannot_reach_context_citations_or_llm_prompt(self):
        chunks = [
            {
                "chunk_id": "allowed-1",
                "document_id": "allowed-doc",
                "filename": "allowed.md",
                "position": 1,
                "chunk_index": 0,
                "page_number": None,
                "content": "Annual leave is fifteen paid days.",
            },
            {
                "chunk_id": "blocked-1",
                "document_id": "blocked-doc",
                "filename": "blocked.md",
                "position": 1,
                "chunk_index": 0,
                "page_number": None,
                "content": "Annual leave UNAUTHORIZED_SECRET_SENTINEL.",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "chunks.json"
            index_path.write_text(json.dumps(chunks), encoding="utf-8")
            with patch(
                "app.services.rag_service.get_readable_document_ids",
                return_value=frozenset({"allowed-doc"}),
            ), patch(
                "app.services.knowledge_base.is_llm_configured",
                return_value=True,
            ), patch(
                "app.services.knowledge_base.chat_completion",
                return_value="Fifteen paid days.",
            ) as completion:
                result = answer_question(
                    "How much annual leave is provided?",
                    2,
                    retrieval_mode="keyword",
                    index_path=index_path,
                    user_id=TEST_USER_ID,
                    min_score=0.0,
                )

        self.assertEqual(
            {context["document_id"] for context in result["contexts"]},
            {"allowed-doc"},
        )
        self.assertEqual(
            {citation["document_id"] for citation in result["citations"]},
            {"allowed-doc"},
        )
        messages = completion.call_args.args[0]
        prompt_text = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("UNAUTHORIZED_SECRET_SENTINEL", prompt_text)

    def test_llm_generation_receives_retrieved_context(self):
        with patch(
            "app.services.knowledge_base.is_llm_configured",
            return_value=True,
        ), patch(
            "app.services.knowledge_base.chat_completion",
            return_value="Employees receive 15 paid annual leave days.",
        ) as completion:
            result = build_answer("How many leave days?", [CONTEXT])

        self.assertEqual(result["mode"], "llm")
        messages = completion.call_args.args[0]
        self.assertIn(CONTEXT["content"], messages[1]["content"])
        self.assertIn("Page Number: 7", messages[1]["content"])

    def test_citations_are_deduplicated_and_preserve_location(self):
        citations = build_citations([CONTEXT, {**CONTEXT, "score": 0.8}])

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["chunk_id"], "policy-1")
        self.assertEqual(citations[0]["chunk_index"], 0)
        self.assertEqual(citations[0]["page_number"], 7)

    def test_resolve_min_score_keeps_explicit_override(self):
        self.assertEqual(resolve_min_score("vector", 0.37), 0.37)

    def test_hybrid_rerank_uses_frozen_component_trace_before_generation(self):
        configuration = RerankedHybridConfig()
        trace = RerankedHybridSearchResult(
            candidates_before_rerank=[{**CONTEXT, "retrieval_mode": "rrf"}],
            results_after_rerank=[
                {**CONTEXT, "retrieval_mode": "hybrid_rerank", "rerank_score": 8.0}
            ],
            configuration=configuration,
        )
        answer = {
            "answer": "Employees receive 15 days.",
            "mode": "llm",
            "model": "qwen3:8b",
            "llm_error": None,
            "citations": [{"chunk_id": "policy-1"}],
        }

        with patch(
            "app.services.rag_service.run_reranked_hybrid_search",
            return_value=trace,
        ) as retrieve, patch(
            "app.services.rag_service.finalize_contexts",
            return_value=trace.results_after_rerank,
        ) as finalize, patch(
            "app.services.rag_service.build_answer",
            return_value=answer,
        ), patch(
            "app.services.rag_service.get_readable_document_ids",
            return_value=frozenset({"policy"}),
        ):
            result = answer_question(
                "How many leave days?",
                2,
                retrieval_mode="hybrid_rerank",
                reranked_hybrid_config=configuration,
                user_id=TEST_USER_ID,
            )

        retrieve.assert_called_once()
        finalize.assert_called_once_with(
            "How many leave days?",
            trace.results_after_rerank,
            None,
            allowed_document_ids=frozenset({"policy"}),
        )
        self.assertEqual(
            result["retrieval_evidence"]["candidates_before_rerank"][0]["retrieval_mode"],
            "rrf",
        )
        self.assertEqual(
            result["retrieval_evidence"]["results_after_rerank"][0]["rerank_score"],
            8.0,
        )


if __name__ == "__main__":
    unittest.main()
