"""Reusable dataset, retrieval, generation, and RAG evaluation helpers."""

from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.direct_prompt_injection import (
    DirectPromptInjectionRunner,
    aggregate_direct_prompt_injection_results,
    classify_direct_prompt_injection_result,
    load_direct_prompt_injection_cases,
)
from app.evaluation.failure_analysis import (
    aggregate_classifications,
    build_failure_analysis_artifact,
    classify_failure_case,
)
from app.evaluation.indirect_prompt_injection import (
    aggregate_indirect_prompt_injection_results,
    build_delivery_evidence,
    classify_model_outcome,
    load_indirect_prompt_injection_cases,
)
from app.evaluation.rag import RAGEvaluationConfig, RAGEvaluationRunner
from app.evaluation.security_comparison import (
    build_security_comparison,
    load_security_evaluation_manifest,
)
from app.evaluation.retrieval import (
    aggregate_retrieval_results,
    calculate_retrieval_metrics,
)
from app.evaluation.unanswerable import (
    UnanswerableEvaluationRunner,
    aggregate_unanswerable_results,
    classify_unanswerable_result,
    load_unanswerable_cases,
)


__all__ = [
    "RAGEvaluationConfig",
    "RAGEvaluationRunner",
    "DirectPromptInjectionRunner",
    "UnanswerableEvaluationRunner",
    "aggregate_classifications",
    "aggregate_direct_prompt_injection_results",
    "aggregate_indirect_prompt_injection_results",
    "aggregate_retrieval_results",
    "aggregate_unanswerable_results",
    "build_failure_analysis_artifact",
    "build_security_comparison",
    "classify_failure_case",
    "classify_direct_prompt_injection_result",
    "build_delivery_evidence",
    "classify_model_outcome",
    "calculate_retrieval_metrics",
    "classify_unanswerable_result",
    "load_evaluation_dataset",
    "load_direct_prompt_injection_cases",
    "load_indirect_prompt_injection_cases",
    "load_security_evaluation_manifest",
    "load_unanswerable_cases",
]
