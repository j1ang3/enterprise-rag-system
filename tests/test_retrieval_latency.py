import json
import unittest
from copy import deepcopy
from pathlib import Path

from app.evaluation.latency import (
    aggregate_latency_samples,
    aggregate_rerank_stage,
    calculate_latency_delta,
    extract_quality_evidence,
    measure_query_methods,
    validate_latency_record,
    validate_quality_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "evals" / "retrieval_evaluation_config.json"
QUALITY_PATH = (
    PROJECT_ROOT / "evals" / "results" / "W7-T3-retrieval-evaluation.json"
)
LATENCY_PATH = (
    PROJECT_ROOT / "evals" / "results" / "W7-T4-latency-trade-off.json"
)


def _candidate(index: int) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": "doc-1",
        "filename": "policy.md",
        "content": f"candidate {index}",
    }


class FakeVector:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        if self.fail:
            raise RuntimeError("vector unavailable")
        return [_candidate(index) for index in range(1, top_k + 1)]


class FakeHybrid:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def retrieve_fused(
        self,
        query: str,
        top_k: int,
        *,
        candidate_depth: int,
        rrf_k: int,
    ) -> list[dict]:
        if self.fail:
            raise RuntimeError("hybrid unavailable")
        return [_candidate(index) for index in range(1, top_k + 1)]


class FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, candidates: list[dict], *, top_k: int) -> list[dict]:
        self.calls += 1
        return list(reversed(candidates))[:top_k]


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class RetrievalLatencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "offline_evaluation_depth": 3,
            "rrf_output_count": 3,
            "per_source_candidate_depth": 5,
            "rrf_k": 60,
        }

    def test_timing_records_have_positive_durations_and_method_identity(self):
        records = self._measure()

        self.assertEqual(
            [record["method"] for record in records],
            ["vector", "hybrid_rrf", "hybrid_reranker"],
        )
        self.assertAlmostEqual(records[0]["total_retrieval_ms"], 1.0)
        self.assertAlmostEqual(records[1]["total_retrieval_ms"], 3.0)
        self.assertAlmostEqual(records[2]["rerank_ms"], 4.0)
        self.assertAlmostEqual(records[2]["total_retrieval_ms"], 7.0)
        for record in records:
            validate_latency_record(record)
            self.assertGreater(record["total_retrieval_ms"], 0)

    def test_hybrid_and_reranker_share_exact_candidate_identity(self):
        records = self._measure()

        self.assertEqual(
            records[2]["pre_rerank_chunk_ids"], records[1]["result_chunk_ids"]
        )
        self.assertEqual(
            records[2]["result_chunk_ids"],
            list(reversed(records[1]["result_chunk_ids"])),
        )

    def test_warmup_and_cold_start_are_excluded_from_aggregate(self):
        cold = self._measure(phase="cold_start", run_id="cold")
        warmup = self._measure(phase="warmup", run_id="warmup")
        measured = self._measure(phase="measured", run_id="measured")

        aggregates = aggregate_latency_samples(cold + warmup + measured)

        self.assertEqual(aggregates["vector"]["sample_count"], 1)
        self.assertEqual(aggregates["vector"]["p50_ms"], 1.0)
        self.assertEqual(aggregates["hybrid_rrf"]["p50_ms"], 3.0)
        self.assertEqual(aggregates["hybrid_reranker"]["p50_ms"], 7.0)

    def test_aggregation_calculates_mean_p50_and_interpolated_p95(self):
        samples = []
        for index, duration in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
            samples.extend(
                self._successful_records(
                    run_id=f"run-{index}",
                    vector_ms=duration,
                    hybrid_ms=duration + 1,
                    rerank_ms=duration + 2,
                )
            )

        aggregates = aggregate_latency_samples(samples)

        self.assertEqual(aggregates["vector"]["sample_count"], 4)
        self.assertEqual(aggregates["vector"]["mean_ms"], 2.5)
        self.assertEqual(aggregates["vector"]["p50_ms"], 2.5)
        self.assertAlmostEqual(aggregates["vector"]["p95_ms"], 3.85)
        self.assertEqual(aggregates["vector"]["min_ms"], 1.0)
        self.assertEqual(aggregates["vector"]["max_ms"], 4.0)

    def test_rerank_stage_is_aggregated_separately(self):
        samples = self._successful_records(
            run_id="run-1", vector_ms=1.0, hybrid_ms=2.0, rerank_ms=5.0
        )

        stage = aggregate_rerank_stage(samples)

        self.assertEqual(stage["sample_count"], 1)
        self.assertEqual(stage["p50_ms"], 5.0)

    def test_failed_sample_is_explicit_and_does_not_pollute_aggregate(self):
        records = measure_query_methods(
            "q001",
            "question",
            run_id="measured-pass-1",
            phase="measured",
            vector_retriever=FakeVector(fail=True),
            hybrid_retriever=FakeHybrid(),
            reranker=FakeReranker(),
            config=self.config,
            clock=FakeClock([0.0, 0.001, 0.002, 0.005, 0.006, 0.010]),
        )

        vector = records[0]
        self.assertEqual(vector["status"], "failed")
        self.assertIsNone(vector["total_retrieval_ms"])
        self.assertEqual(vector["error"]["type"], "RuntimeError")
        aggregates = aggregate_latency_samples(records)
        self.assertEqual(aggregates["vector"]["sample_count"], 0)
        self.assertEqual(aggregates["vector"]["failed_sample_count"], 1)
        self.assertIsNone(aggregates["vector"]["mean_ms"])
        self.assertEqual(aggregates["hybrid_rrf"]["sample_count"], 1)

    def test_hybrid_failure_marks_reranker_failed_without_calling_it(self):
        reranker = FakeReranker()
        records = measure_query_methods(
            "q001",
            "question",
            run_id="measured-pass-1",
            phase="measured",
            vector_retriever=FakeVector(),
            hybrid_retriever=FakeHybrid(fail=True),
            reranker=reranker,
            config=self.config,
            clock=FakeClock([0.0, 0.001, 0.002, 0.005]),
        )

        self.assertEqual(records[1]["status"], "failed")
        self.assertEqual(records[2]["status"], "failed")
        self.assertEqual(reranker.calls, 0)

    def test_latency_delta_handles_absolute_relative_and_zero_baseline(self):
        delta = calculate_latency_delta(
            {"mean_ms": 4.0, "p50_ms": 5.0, "p95_ms": 8.0},
            {"mean_ms": 6.0, "p50_ms": 8.0, "p95_ms": 12.0},
        )
        zero = calculate_latency_delta(
            {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
            {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0},
        )

        self.assertEqual(delta["p50_ms"]["absolute_delta_ms"], 3.0)
        self.assertEqual(delta["p50_ms"]["relative_delta_percent"], 60.0)
        self.assertIsNone(zero["p50_ms"]["relative_delta_percent"])

    def test_real_quality_artifact_matches_frozen_configuration_identity(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

        validate_quality_artifact(
            quality,
            frozen_manifest=manifest,
            frozen_manifest_sha256=quality["frozen_manifest"]["sha256"],
        )
        evidence = extract_quality_evidence(quality, primary_k=2)

        self.assertEqual(evidence["primary_k"], 2)
        self.assertEqual(
            evidence["quality_deltas"]["hybrid_to_hybrid_reranker"],
            {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0},
        )
        self.assertEqual(
            evidence["reranker_effect_summary"]["ordering_changed_query_ids"],
            ["q005", "q015"],
        )

    def test_quality_artifact_configuration_drift_is_rejected(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
        drifted = deepcopy(quality)
        drifted["resolved_configuration"]["reranker_candidate_count"] = 5

        with self.assertRaisesRegex(ValueError, "configuration differs"):
            validate_quality_artifact(
                drifted,
                frozen_manifest=manifest,
                frozen_manifest_sha256=quality["frozen_manifest"]["sha256"],
            )

    def test_real_latency_artifact_preserves_config_and_sample_identity(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        artifact = json.loads(LATENCY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(artifact["task"], "W7-T4")
        self.assertEqual(artifact["status"], "complete")
        self.assertFalse(
            artifact["configuration_identity"]["configuration_changed_for_latency"]
        )
        self.assertEqual(
            artifact["configuration_identity"]["resolved_configuration"],
            manifest["resolved_configuration"],
        )
        self.assertTrue(artifact["warmup"]["excluded_from_aggregate"])
        self.assertEqual(len(artifact["per_sample_results"]), 75)
        self.assertEqual(
            {sample["method"] for sample in artifact["per_sample_results"]},
            {"vector", "hybrid_rrf", "hybrid_reranker"},
        )
        self.assertTrue(
            all(
                sample["phase"] == "measured"
                for sample in artifact["per_sample_results"]
            )
        )
        for method in ("vector", "hybrid_rrf", "hybrid_reranker"):
            self.assertEqual(
                artifact["aggregate_results"][method]["sample_count"], 25
            )
            self.assertEqual(
                artifact["aggregate_results"][method]["failed_sample_count"], 0
            )
        self.assertEqual(
            artifact["reranker_stage_latency"]["sample_count"], 25
        )

    def test_unknown_method_identity_is_rejected(self):
        record = self._successful_records(
            run_id="run-1", vector_ms=1.0, hybrid_ms=2.0, rerank_ms=3.0
        )[0]
        record["method"] = "unknown"

        with self.assertRaisesRegex(ValueError, "unknown latency method"):
            validate_latency_record(record)

    def _measure(
        self,
        *,
        phase: str = "measured",
        run_id: str = "measured-pass-1",
    ) -> list[dict]:
        return measure_query_methods(
            "q001",
            "question",
            run_id=run_id,
            phase=phase,
            vector_retriever=FakeVector(),
            hybrid_retriever=FakeHybrid(),
            reranker=FakeReranker(),
            config=self.config,
            clock=FakeClock([0.0, 0.001, 0.002, 0.005, 0.006, 0.010]),
        )

    @staticmethod
    def _successful_records(
        *,
        run_id: str,
        vector_ms: float,
        hybrid_ms: float,
        rerank_ms: float,
    ) -> list[dict]:
        common = {
            "query_id": "q001",
            "run_id": run_id,
            "phase": "measured",
            "status": "success",
            "error": None,
            "pre_rerank_chunk_ids": None,
        }
        return [
            {
                **common,
                "method": "vector",
                "total_retrieval_ms": vector_ms,
                "rerank_ms": None,
                "stage_latency_ms": {"vector_ms": vector_ms},
                "result_chunk_ids": ["chunk-1"],
            },
            {
                **common,
                "method": "hybrid_rrf",
                "total_retrieval_ms": hybrid_ms,
                "rerank_ms": None,
                "stage_latency_ms": {
                    "hybrid_retrieval_and_fusion_ms": hybrid_ms
                },
                "result_chunk_ids": ["chunk-1"],
            },
            {
                **common,
                "method": "hybrid_reranker",
                "total_retrieval_ms": hybrid_ms + rerank_ms,
                "rerank_ms": rerank_ms,
                "stage_latency_ms": {
                    "shared_hybrid_retrieval_and_fusion_ms": hybrid_ms,
                    "rerank_ms": rerank_ms,
                },
                "result_chunk_ids": ["chunk-1"],
                "pre_rerank_chunk_ids": ["chunk-1"],
            },
        ]


if __name__ == "__main__":
    unittest.main()
