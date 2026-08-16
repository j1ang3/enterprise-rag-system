import unittest

from app.retrieval.reranker import (
    CrossEncoderScorer,
    RerankerModelError,
    SemanticReranker,
)


def candidate(chunk_id: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "filename": f"{chunk_id}.md",
        "position": 1,
        "chunk_index": 0,
        "page_number": 3,
        "content": content,
        "text": content,
        "metadata": {"department": "engineering"},
        "source": "rrf",
        "retrieval_mode": "rrf",
        "score": 0.032,
        "fused_score": 0.032,
        "matched_sources": ["vector", "bm25"],
        "source_ranks": {"vector": 2, "bm25": 1},
        "source_scores": {"vector": 0.81, "bm25": 5.4},
    }


class FakeScorer:
    def __init__(self, scores) -> None:
        self.scores = scores
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.append(list(pairs))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


class SemanticRerankerTests(unittest.TestCase):
    def test_reorders_candidates_and_adds_rerank_score(self):
        candidates = [
            candidate("a", "alpha"),
            candidate("b", "beta"),
            candidate("c", "gamma"),
        ]

        results = SemanticReranker(FakeScorer([0.2, 0.9, 0.5])).rerank(
            "policy query",
            candidates,
        )

        self.assertEqual([item["chunk_id"] for item in results], ["b", "c", "a"])
        self.assertEqual(
            {item["chunk_id"]: item["rerank_score"] for item in results},
            {"a": 0.2, "b": 0.9, "c": 0.5},
        )

    def test_preserves_existing_fields_without_mutating_candidates(self):
        original = candidate("a", "retrieval augmented generation")

        result = SemanticReranker(FakeScorer([0.8])).rerank(
            "What is RAG?",
            [original],
        )[0]

        for field in (
            "chunk_id",
            "document_id",
            "filename",
            "page_number",
            "metadata",
            "score",
            "fused_score",
            "matched_sources",
            "source_ranks",
            "source_scores",
            "retrieval_mode",
        ):
            self.assertEqual(result[field], original[field])
        self.assertNotIn("rerank_score", original)
        self.assertIsNot(result["metadata"], original["metadata"])

    def test_top_k_is_applied_after_reranking(self):
        candidates = [candidate(str(index), str(index)) for index in range(5)]

        results = SemanticReranker(FakeScorer([0.2, 0.9, 0.5, 0.8, 0.1])).rerank(
            "query",
            candidates,
            top_k=3,
        )

        self.assertEqual([item["chunk_id"] for item in results], ["1", "3", "2"])

    def test_empty_and_single_candidate_lists(self):
        empty_scorer = FakeScorer([])
        reranker = SemanticReranker(empty_scorer)

        self.assertEqual(reranker.rerank("query", []), [])
        self.assertEqual(empty_scorer.calls, [])

        result = SemanticReranker(FakeScorer([0.4])).rerank(
            "query",
            [candidate("only", "only content")],
        )
        self.assertEqual(result[0]["chunk_id"], "only")
        self.assertEqual(result[0]["rerank_score"], 0.4)

    def test_rejects_empty_query_and_invalid_top_k(self):
        reranker = SemanticReranker(FakeScorer([]))

        for query in ("", "   "):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, "query"):
                reranker.rerank(query, [])

        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "top_k"):
                reranker.rerank("query", [], top_k=top_k)

    def test_top_k_larger_than_candidates_returns_all(self):
        results = SemanticReranker(FakeScorer([0.1, 0.2])).rerank(
            "query",
            [candidate("a", "a"), candidate("b", "b")],
            top_k=10,
        )

        self.assertEqual([item["chunk_id"] for item in results], ["b", "a"])

    def test_equal_scores_keep_original_candidate_order(self):
        candidates = [candidate("a", "a"), candidate("b", "b")]

        results = SemanticReranker(FakeScorer([0.5, 0.5])).rerank(
            "query",
            candidates,
        )

        self.assertEqual([item["chunk_id"] for item in results], ["a", "b"])

    def test_scorer_receives_query_candidate_text_pairs(self):
        scorer = FakeScorer([0.3, 0.7])

        SemanticReranker(scorer).rerank(
            "What is RAG?",
            [
                candidate("a", "RAG retrieves knowledge."),
                {**candidate("b", "unused"), "content": None, "text": "FastAPI."},
            ],
        )

        self.assertEqual(
            scorer.calls,
            [[
                ("What is RAG?", "RAG retrieves knowledge."),
                ("What is RAG?", "FastAPI."),
            ]],
        )

    def test_rejects_missing_text_and_duplicate_chunk_id(self):
        missing_text = candidate("missing", "content")
        missing_text["content"] = None
        missing_text.pop("text")

        with self.assertRaisesRegex(ValueError, "content or text"):
            SemanticReranker(FakeScorer([])).rerank("query", [missing_text])

        duplicate = candidate("duplicate", "content")
        with self.assertRaisesRegex(ValueError, "duplicate candidate chunk_id"):
            SemanticReranker(FakeScorer([])).rerank(
                "query",
                [duplicate, duplicate],
            )

    def test_model_scoring_error_and_score_length_mismatch_fail_clearly(self):
        with self.assertRaisesRegex(RerankerModelError, "scoring failed"):
            SemanticReranker(FakeScorer(RuntimeError("provider detail"))).rerank(
                "query",
                [candidate("a", "a")],
            )

        with self.assertRaisesRegex(RerankerModelError, "different number"):
            SemanticReranker(FakeScorer([0.1])).rerank(
                "query",
                [candidate("a", "a"), candidate("b", "b")],
            )

    def test_cross_encoder_model_is_loaded_once_and_reused(self):
        class FakeModel:
            def __init__(self) -> None:
                self.calls = []

            def predict(self, pairs):
                self.calls.append(pairs)
                return [0.6] * len(pairs)

        model = FakeModel()
        loader_calls = []

        def loader(model_name, *, local_files_only):
            loader_calls.append((model_name, local_files_only))
            return model

        scorer = CrossEncoderScorer(
            "test-model",
            local_files_only=True,
            model_loader=loader,
        )
        reranker = SemanticReranker(scorer)

        reranker.rerank("first", [candidate("a", "a")])
        reranker.rerank("second", [candidate("b", "b")])

        self.assertEqual(loader_calls, [("test-model", True)])
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
