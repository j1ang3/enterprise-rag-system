"""Small, explainable security layers around the existing RAG pipeline.

The signal detector is deliberately observe-only. It never removes, rewrites, trusts,
or reorders retrieved content. Deterministic output checks block only explicit policy
violations that the application can verify without guessing model intent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


BASELINE_SECURITY_MODE = "baseline"
LAYERED_SECURITY_MODE = "layered"
BASELINE_SECURITY_POLICY_VERSION = "w9-t2-t3-baseline.v1"
LAYERED_SECURITY_POLICY_VERSION = "w9-t4-layered-defenses.v1"

SAFE_BLOCKED_RESPONSE = (
    "I cannot provide that response because it did not satisfy the "
    "application's safety checks."
)


@dataclass(frozen=True)
class DefenseDefinition:
    defense_id: str
    name: str
    layer: str
    threat_ids: tuple[str, ...]
    behavior: str
    limitation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "defense_id": self.defense_id,
            "name": self.name,
            "layer": self.layer,
            "threat_ids": list(self.threat_ids),
            "behavior": self.behavior,
            "limitation": self.limitation,
        }


DEFENSE_REGISTRY = (
    DefenseDefinition(
        defense_id="DEF-PROMPT-001",
        name="trusted_instruction_hierarchy",
        layer="prompt",
        threat_ids=("DPI-001", "IPI-001", "SPL-001"),
        behavior="Separates application instructions from untrusted requests and evidence.",
        limitation="Model adherence remains probabilistic.",
    ),
    DefenseDefinition(
        defense_id="DEF-CONTEXT-001",
        name="untrusted_context_framing",
        layer="context",
        threat_ids=("IPI-001", "MD-001", "SIL-001"),
        behavior="Frames complete retrieved chunks as untrusted data without deleting text.",
        limitation="Delimiters are model-visible text, not an authorization boundary.",
    ),
    DefenseDefinition(
        defense_id="DEF-SIGNAL-001",
        name="instruction_like_context_signal",
        layer="input_observability",
        threat_ids=("IPI-001", "MD-001", "LOG-001"),
        behavior="Marks instruction-like context features for diagnostics only.",
        limitation="Heuristics can miss attacks and flag legitimate security discussions.",
    ),
    DefenseDefinition(
        defense_id="DEF-OUTPUT-001",
        name="deterministic_output_validation",
        layer="output",
        threat_ids=("SPL-001", "SIL-001", "CI-001"),
        behavior="Fails closed for protected canary, prompt-clause, or citation-contract violations.",
        limitation="Cannot recognize every paraphrased leak or every unsupported factual claim.",
    ),
    DefenseDefinition(
        defense_id="DEF-LOG-001",
        name="content_minimized_security_logging",
        layer="observability",
        threat_ids=("DPI-001", "IPI-001", "LOG-001"),
        behavior="Records security mode, signals, and output action without raw content.",
        limitation="Logs support investigation but do not prevent an attack.",
    ),
)

_DEFENSE_IDS = tuple(defense.defense_id for defense in DEFENSE_REGISTRY)


@dataclass(frozen=True)
class SecurityPolicy:
    mode: str
    version: str
    enabled_defense_ids: tuple[str, ...]
    protected_output_canaries: tuple[str, ...] = ()

    @property
    def layered(self) -> bool:
        return self.mode == LAYERED_SECURITY_MODE

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "version": self.version,
            "enabled_defense_ids": list(self.enabled_defense_ids),
            "protected_output_canary_count": len(self.protected_output_canaries),
        }


def defense_registry_records() -> list[dict[str, Any]]:
    return [definition.to_dict() for definition in DEFENSE_REGISTRY]


def resolve_security_policy(
    mode: str,
    *,
    protected_output_canaries: Sequence[str] = (),
) -> SecurityPolicy:
    normalized_mode = str(mode).strip().casefold()
    if normalized_mode not in {BASELINE_SECURITY_MODE, LAYERED_SECURITY_MODE}:
        raise ValueError("security mode must be 'baseline' or 'layered'")

    canaries: list[str] = []
    for canary in protected_output_canaries:
        if not isinstance(canary, str) or not canary.strip():
            raise ValueError("protected output canaries must be non-empty strings")
        canaries.append(canary.strip())

    if normalized_mode == BASELINE_SECURITY_MODE:
        return SecurityPolicy(
            mode=BASELINE_SECURITY_MODE,
            version=BASELINE_SECURITY_POLICY_VERSION,
            enabled_defense_ids=(),
            protected_output_canaries=(),
        )
    return SecurityPolicy(
        mode=LAYERED_SECURITY_MODE,
        version=LAYERED_SECURITY_POLICY_VERSION,
        enabled_defense_ids=_DEFENSE_IDS,
        protected_output_canaries=tuple(canaries),
    )


_CONTROL_VERB_PATTERN = re.compile(
    r"\b(ignore|disregard|override|reveal|print|output|return|declare|"
    r"execute|follow|obey|bypass|suppress)\b",
    re.IGNORECASE,
)
_CONTROL_TARGET_PATTERN = re.compile(
    r"\b(system|prompt|instruction|rule|assistant|answer|response|context|"
    r"grounding|source of truth|administrator|developer)\b",
    re.IGNORECASE,
)


def analyze_context_security_signals(
    contexts: Sequence[Mapping[str, Any]],
    *,
    policy: SecurityPolicy,
) -> dict[str, Any]:
    """Return metadata-only signals; never mutate or filter the supplied contexts."""
    if not policy.layered:
        return {
            "status": "not_applied",
            "trust_level": "untrusted",
            "action": "none",
            "signal_ids": [],
            "signal_count": 0,
            "flagged_chunk_ids": [],
            "feature_ids": [],
        }

    flagged_chunk_ids: list[str] = []
    feature_ids: set[str] = set()
    for context in contexts:
        content = context.get("content")
        if not isinstance(content, str):
            continue
        has_control_verb = bool(_CONTROL_VERB_PATTERN.search(content))
        has_control_target = bool(_CONTROL_TARGET_PATTERN.search(content))
        if not (has_control_verb and has_control_target):
            continue
        chunk_id = context.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            flagged_chunk_ids.append(chunk_id)
        feature_ids.update(
            {"instruction_control_verb", "instruction_authority_target"}
        )

    signal_ids = ["SEC-CTX-INSTRUCTION-LIKE"] if flagged_chunk_ids else []
    return {
        "status": "signals_detected" if signal_ids else "no_signals_detected",
        "trust_level": "untrusted",
        "action": "observe_only",
        "signal_ids": signal_ids,
        "signal_count": len(flagged_chunk_ids),
        "flagged_chunk_ids": list(dict.fromkeys(flagged_chunk_ids)),
        "feature_ids": sorted(feature_ids),
    }


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _blocked_result(
    answer_result: Mapping[str, Any],
    *,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    secured = dict(answer_result)
    secured["answer"] = SAFE_BLOCKED_RESPONSE
    secured["citations"] = []
    return secured, {
        "status": "blocked",
        "blocked": True,
        "blocked_reason": reason,
        "blocking_defense_id": "DEF-OUTPUT-001",
        "matched_prompt_fragment_count": 0,
        "protected_output_canary_matched": reason == "protected_output_canary",
    }


def _citation_contract_is_valid(
    citations: object,
    contexts: Sequence[Mapping[str, Any]],
) -> bool:
    if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
        return False
    allowed_ids = {
        context.get("chunk_id")
        for context in contexts
        if isinstance(context.get("chunk_id"), str) and context.get("chunk_id")
    }
    for citation in citations:
        if not isinstance(citation, Mapping):
            return False
        chunk_id = citation.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id not in allowed_ids:
            return False
    return True


def validate_and_secure_output(
    answer_result: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    *,
    policy: SecurityPolicy,
    protected_prompt_fragments: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply deterministic output checks and return a safe replacement on violation."""
    if not policy.layered:
        return dict(answer_result), {
            "status": "not_applied",
            "blocked": False,
            "blocked_reason": None,
            "blocking_defense_id": None,
            "matched_prompt_fragment_count": 0,
            "protected_output_canary_matched": False,
        }

    answer = answer_result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return _blocked_result(answer_result, reason="invalid_answer_contract")

    if answer_result.get("mode") == "local_fallback":
        return _blocked_result(answer_result, reason="unsafe_local_fallback")

    normalized_answer = _normalized_text(answer)
    for canary in policy.protected_output_canaries:
        if _normalized_text(canary) in normalized_answer:
            return _blocked_result(answer_result, reason="protected_output_canary")

    matched_fragments = {
        _normalized_text(fragment)
        for fragment in protected_prompt_fragments
        if isinstance(fragment, str)
        and fragment.strip()
        and _normalized_text(fragment) in normalized_answer
    }
    if len(matched_fragments) >= 2:
        secured, validation = _blocked_result(
            answer_result,
            reason="protected_prompt_instructions",
        )
        validation["matched_prompt_fragment_count"] = len(matched_fragments)
        return secured, validation

    if not _citation_contract_is_valid(answer_result.get("citations"), contexts):
        return _blocked_result(answer_result, reason="invalid_citation_contract")

    return dict(answer_result), {
        "status": "passed",
        "blocked": False,
        "blocked_reason": None,
        "blocking_defense_id": None,
        "matched_prompt_fragment_count": len(matched_fragments),
        "protected_output_canary_matched": False,
    }
