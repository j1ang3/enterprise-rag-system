import unittest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever


def candidate(
    chunk_id: str,
    score: float,
    *,
    page_number: int | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "filename": f"{chunk_id}.txt",
        "position": 1,
        "chunk_index": 0,
        "page_number": page_number,
        "content": f"content for {chunk_id}",
        "text": f"content for {chunk_id}",
        "metadata": metadata or {"document_id": f"doc-{chunk_id}"},
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


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_formula_uses_one_based_rank_and_accumulates_sources(self):
        vector_results = [candidate("a", 0.95), candidate("shared", 0.85)]
        bm25_results = [candidate("shared", 8.2), candidate("c", 5.1)]

        results = reciprocal_rank_fusion(
            (("vector", vector_results), ("bm25", bm25_results)),
            top_k=3,
            rrf_k=60,
        )

        shared = results[0]
        expected_score = 1 / (60 + 2) + 1 / (60 + 1)
        self.assertAlmostEqual(shared["fused_score"], expected_score)
        self.assertAlmostEqual(
            next(item for item in results if item["chunk_id"] == "a")["fused_score"],
            1 / (60 + 1),
        )
        self.assertEqual([item["chunk_id"] for item in results], ["shared", "a", "c"])

    def test_duplicate_chunk_is_returned_once_with_source_details(self):
        vector = candidate("shared", 0.83, page_number=3)
        bm25 = candidate("shared", 7.41, metadata={"department": "people"})

        results = reciprocal_rank_fusion(
            (("vector", [vector]), ("bm25", [bm25])),
            top_k=5,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["matched_sources"], ["vector", "bm25"])
        self.assertEqual(result["source_ranks"], {"vector": 1, "bm25": 1})
        self.assertEqual(result["source_scores"], {"vector": 0.83, "bm25": 7.41})
        self.assertEqual(result["metadata"]["document_id"], "doc-shared")
        self.assertEqual(result["metadata"]["department"], "people")
        self.assertEqual(result["page_number"], 3)
        self.assertEqual(result["source"], "rrf")
        self.assertEqual(result["retrieval_mode"], "rrf")

    def test_top_k_is_applied_after_fusion_and_deduplication(self):
        results = reciprocal_rank_fusion(
            (
                ("vector", [candidate("a", 0.9), candidate("shared", 0.8)]),
                ("bm25", [candidate("shared", 9.0), candidate("b", 5.0)]),
            ),
            top_k=1,
        )

        self.assertEqual([item["chunk_id"] for item in results], ["shared"])

    def test_single_source_and_empty_inputs_are_supported(self):
        cases = (
            ((("vector", [candidate("vector-only", 0.8)]), ("bm25", [])), "vector"),
            ((("vector", []), ("bm25", [candidate("bm25-only", 4.0)])), "bm25"),
        )

        for ranked_results, expected_source in cases:
            with self.subTest(expected_source=expected_source):
                results = reciprocal_rank_fusion(ranked_results, top_k=3)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["matched_sources"], [expected_source])

        self.assertEqual(
            reciprocal_rank_fusion((("vector", []), ("bm25", [])), top_k=3),
            [],
        )
        self.assertEqual(reciprocal_rank_fusion((), top_k=3), [])

    def test_tie_break_uses_best_rank_then_first_seen_order(self):
        ranked_results = (
            ("vector", [candidate("first", 0.8)]),
            ("bm25", [candidate("second", 8.0)]),
        )

        first_run = reciprocal_rank_fusion(ranked_results, top_k=2)
        second_run = reciprocal_rank_fusion(ranked_results, top_k=2)

        self.assertEqual(
            [item["chunk_id"] for item in first_run],
            ["first", "second"],
        )
        self.assertEqual(
            [item["chunk_id"] for item in first_run],
            [item["chunk_id"] for item in second_run],
        )

    def test_duplicate_within_one_source_contributes_only_best_rank(self):
        duplicate = candidate("duplicate", 0.9)

        result = reciprocal_rank_fusion(
            (("vector", [duplicate, duplicate]),),
            top_k=1,
        )[0]

        self.assertAlmostEqual(result["fused_score"], 1 / 61)
        self.assertEqual(result["source_ranks"], {"vector": 1})

    def test_invalid_parameters_and_missing_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "rrf_k"):
            reciprocal_rank_fusion((), top_k=1, rrf_k=0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            reciprocal_rank_fusion((), top_k=0)

        invalid = candidate("invalid", 0.5)
        invalid.pop("chunk_id")
        with self.assertRaisesRegex(ValueError, "chunk_id"):
            reciprocal_rank_fusion((("vector", [invalid]),), top_k=1)

    def test_hybrid_retriever_fuses_full_candidate_depth(self):
        vector = StubRetriever(
            "vector",
            [candidate("a", 0.9), candidate("shared", 0.8), candidate("v3", 0.7)],
        )
        bm25 = StubRetriever(
            "bm25",
            [candidate("shared", 8.0), candidate("b", 6.0), candidate("b3", 4.0)],
        )

        results = HybridRetriever(
            (vector, bm25),
            allowed_document_ids={
                item["document_id"] for item in vector.results + bm25.results
            },
        ).retrieve_fused(
            "policy",
            top_k=2,
            candidate_depth=3,
        )

        self.assertEqual(vector.calls, [("policy", 3)])
        self.assertEqual(bm25.calls, [("policy", 3)])
        self.assertEqual([item["chunk_id"] for item in results], ["shared", "a"])


if __name__ == "__main__":
    unittest.main()
