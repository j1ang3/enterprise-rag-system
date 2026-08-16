import unittest
from unittest.mock import patch

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import VectorRetriever
from app.services.search_service import (
    RerankedHybridConfig,
    run_reranked_hybrid_search,
)


def result(chunk_id: str, content: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "filename": f"{chunk_id}.txt",
        "position": 1,
        "chunk_index": 0,
        "page_number": None,
        "content": content,
        "score": score,
    }


class StubRetriever:
    def __init__(self, source: str, results: list[dict]) -> None:
        self.source = source
        self.results = results
        self.calls = []

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class StubHybrid:
    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates
        self.calls = []

    def retrieve_fused(self, query, top_k, *, candidate_depth, rrf_k):
        self.calls.append((query, top_k, candidate_depth, rrf_k))
        return self.candidates[:top_k]


class StubReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query, candidates, top_k=None):
        self.calls.append((query, candidates, top_k))
        return [{**candidate, "rerank_score": float(index)} for index, candidate in enumerate(reversed(candidates), start=1)][:top_k]


class HybridRetrieverTests(unittest.TestCase):
    def test_calls_vector_and_bm25_retrievers(self):
        vector = StubRetriever("vector", [result("vector-1", "semantic", 0.91)])
        bm25 = StubRetriever("bm25", [result("bm25-1", "lexical", 2.4)])

        HybridRetriever(
            (vector, bm25),
            allowed_document_ids={"doc-vector-1", "doc-bm25-1"},
        ).retrieve("annual leave", top_k=3)

        self.assertEqual(vector.calls, [("annual leave", 3)])
        self.assertEqual(bm25.calls, [("annual leave", 3)])

    def test_normalizes_results_from_different_sources(self):
        vector_result = result("vector-1", "semantic match", 0.91)
        bm25_result = {
            **result("bm25-1", "exact match", 2.4),
            "metadata": {"department": "people"},
            "bm25_score": 2.4,
        }

        results = HybridRetriever(
            (
                StubRetriever("vector", [vector_result]),
                StubRetriever("bm25", [bm25_result]),
            ),
            allowed_document_ids={"doc-vector-1", "doc-bm25-1"},
        ).retrieve("leave", top_k=1)

        self.assertEqual([item["source"] for item in results], ["vector", "bm25"])
        self.assertTrue(all(item["text"] == item["content"] for item in results))
        self.assertEqual(results[0]["metadata"]["document_id"], "doc-vector-1")
        self.assertEqual(results[1]["metadata"]["department"], "people")
        self.assertEqual(results[0]["score"], 0.91)
        self.assertEqual(results[1]["bm25_score"], 2.4)

    def test_handles_one_or_both_sources_returning_empty_results(self):
        cases = (
            ([], [result("bm25-1", "lexical", 2.0)], ["bm25"]),
            ([result("vector-1", "semantic", 0.8)], [], ["vector"]),
            ([], [], []),
        )

        for vector_results, bm25_results, expected_sources in cases:
            with self.subTest(expected_sources=expected_sources):
                results = HybridRetriever(
                    (
                        StubRetriever("vector", vector_results),
                        StubRetriever("bm25", bm25_results),
                    ),
                    allowed_document_ids={
                        item["document_id"]
                        for item in vector_results + bm25_results
                    },
                ).retrieve("policy", top_k=2)

                self.assertEqual(
                    [item["source"] for item in results],
                    expected_sources,
                )

    def test_top_k_is_applied_per_source_without_final_ranking(self):
        vector = StubRetriever(
            "vector",
            [result(f"vector-{index}", "semantic", 1.0 - index / 10) for index in range(3)],
        )
        bm25 = StubRetriever(
            "bm25",
            [result(f"bm25-{index}", "lexical", 3.0 - index) for index in range(3)],
        )

        results = HybridRetriever(
            (vector, bm25),
            allowed_document_ids={
                item["document_id"] for item in vector.results + bm25.results
            },
        ).retrieve("policy", top_k=2)

        self.assertEqual(len(results), 4)
        self.assertEqual(
            [item["chunk_id"] for item in results],
            ["vector-0", "vector-1", "bm25-0", "bm25-1"],
        )

    def test_rejects_invalid_top_k(self):
        retriever = HybridRetriever(
            (StubRetriever("vector", []),),
            allowed_document_ids=set(),
        )

        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.retrieve("policy", top_k=0)

    def test_unauthorized_and_missing_identity_candidates_never_enter_rrf(self):
        allowed = result("allowed", "authorized", 0.30)
        blocked = result("blocked", "strong unauthorized match", 0.99)
        missing_identity = {
            key: value
            for key, value in result("missing", "unknown identity", 0.98).items()
            if key != "document_id"
        }
        vector = StubRetriever("vector", [blocked, missing_identity, allowed])
        bm25 = StubRetriever("bm25", [blocked, allowed])

        fused = HybridRetriever(
            (vector, bm25),
            allowed_document_ids={allowed["document_id"]},
        ).retrieve_fused(
            "policy",
            top_k=3,
            candidate_depth=3,
            rrf_k=60,
        )

        self.assertEqual([item["chunk_id"] for item in fused], ["allowed"])
        self.assertEqual(fused[0]["document_id"], allowed["document_id"])

    def test_vector_adapter_uses_existing_vector_search(self):
        expected = [result("vector-1", "semantic", 0.9)]

        with patch(
            "app.retrieval.vector.search_vector_chunks",
            return_value=expected,
        ) as search:
            results = VectorRetriever(
                min_score=0.0,
                allowed_document_ids={"doc-vector-1"},
            ).retrieve("policy", top_k=2)

        self.assertEqual(results, expected)
        search.assert_called_once_with(
            "policy",
            top_k=2,
            index_path=None,
            min_score=0.0,
            allowed_document_ids=frozenset({"doc-vector-1"}),
        )

    def test_reranked_hybrid_service_propagates_frozen_budgets_and_trace(self):
        candidates = [
            result("a", "first", 0.03),
            result("b", "second", 0.02),
            result("c", "third", 0.01),
        ]
        hybrid = StubHybrid(candidates)
        reranker = StubReranker()
        configuration = RerankedHybridConfig()

        with patch(
            "app.services.search_service.build_hybrid_retriever",
            return_value=hybrid,
        ):
            trace = run_reranked_hybrid_search(
                "policy",
                configuration=configuration,
                reranker=reranker,
                allowed_document_ids={"doc-a", "doc-b", "doc-c"},
            )

        self.assertEqual(hybrid.calls, [("policy", 3, 5, 60)])
        self.assertEqual(reranker.calls[0][2], 2)
        self.assertEqual(trace.candidates_before_rerank, candidates)
        self.assertEqual(len(trace.results_after_rerank), 2)
        self.assertTrue(
            all(
                item["retrieval_mode"] == "hybrid_rerank"
                for item in trace.results_after_rerank
            )
        )

    def test_reranker_receives_only_authorized_candidates_from_faulty_hybrid(self):
        allowed = result("allowed", "authorized", 0.1)
        blocked = result("blocked", "unauthorized sentinel", 0.99)
        missing = {
            key: value
            for key, value in result("missing", "missing identity", 0.98).items()
            if key != "document_id"
        }
        hybrid = StubHybrid([blocked, missing, allowed])
        reranker = StubReranker()

        with patch(
            "app.services.search_service.build_hybrid_retriever",
            return_value=hybrid,
        ):
            trace = run_reranked_hybrid_search(
                "policy",
                configuration=RerankedHybridConfig(
                    per_source_candidate_depth=3,
                    rerank_candidate_count=3,
                    final_top_k=1,
                ),
                reranker=reranker,
                allowed_document_ids={allowed["document_id"]},
            )

        self.assertEqual(trace.candidates_before_rerank, [allowed])
        self.assertEqual(reranker.calls[0][1], [allowed])


if __name__ == "__main__":
    unittest.main()
