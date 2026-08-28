"""Optional semantic fallback for meanings that deterministic rules cannot resolve."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .intent import IntentResult
from .model import ShoppingState, StateUpdate


CONTEXT_DEPENDENCY_RE = re.compile(
    r"\b(?:it|that|those|them|the last one|the previous one|same one|similar one)\b",
    re.IGNORECASE,
)
SEMANTIC_COMPARISON_RE = re.compile(
    r"\b(?:more|less|better|similar|same|another|something|not too|kind of|sort of)\b",
    re.IGNORECASE,
)
ALTERNATIVE_RE = re.compile(r"\b(?:or|either|whichever|maybe)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SemanticPolicy:
    """Thresholds controlling when the optional resolver may be called."""

    intent_confidence_threshold: float = 0.70
    recent_history_turns: int = 4


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    should_resolve: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticRequest:
    """Compact, provider-neutral request sent to a semantic resolver."""

    message: str
    current_state: dict[str, Any]
    recent_history: tuple[str, ...]
    asked_attribute: str | None
    rule_intent: str
    rule_intent_confidence: float
    rule_evidence: tuple[str, ...]
    fallback_reasons: tuple[str, ...]


@dataclass(slots=True)
class SemanticResolution:
    """Structured semantic result; free-form model text never enters state."""

    update: StateUpdate = field(default_factory=StateUpdate)
    confidence: float = 0.0


class SemanticResolver(Protocol):
    """Implemented by an LLM adapter or a deterministic test double."""

    def resolve(self, request: SemanticRequest) -> SemanticResolution | None: ...


def _has_attributes(update: StateUpdate) -> bool:
    return bool(
        update.hard_constraint
        or update.soft_constraint
        or update.no_preference
        or update.rejected_values
        or update.boundary_attributes
    )


def decide_semantic_fallback(
    message: str,
    state: ShoppingState,
    intent: IntentResult,
    rule_update: StateUpdate,
    *,
    asked_attribute: str | None = None,
    policy: SemanticPolicy | None = None,
) -> SemanticDecision:
    """Allow semantic resolution only for unresolved or context-dependent input."""

    selected_policy = policy or SemanticPolicy()
    text = " ".join(message.split())
    if not text:
        return SemanticDecision(False)

    reasons: list[str] = []
    has_attributes = _has_attributes(rule_update)
    if intent.confidence < selected_policy.intent_confidence_threshold and not has_attributes:
        reasons.append("low_intent_confidence")
    if intent.is_override and not has_attributes:
        reasons.append("unresolved_override")
    if CONTEXT_DEPENDENCY_RE.search(text) and not asked_attribute:
        reasons.append("context_dependent_reference")
    if SEMANTIC_COMPARISON_RE.search(text) and not has_attributes:
        reasons.append("unresolved_semantic_comparison")
    if ALTERNATIVE_RE.search(text) and not has_attributes:
        reasons.append("unresolved_alternative")

    # A direct answer is already grounded by the previous asked attribute and
    # should stay on the deterministic path whenever rules captured it.
    if asked_attribute and has_attributes:
        return SemanticDecision(False)
    return SemanticDecision(bool(reasons), tuple(dict.fromkeys(reasons)))


def build_semantic_request(
    message: str,
    state: ShoppingState,
    intent: IntentResult,
    decision: SemanticDecision,
    *,
    asked_attribute: str | None = None,
    policy: SemanticPolicy | None = None,
) -> SemanticRequest:
    selected_policy = policy or SemanticPolicy()
    return SemanticRequest(
        message=message,
        current_state=state.to_dict(),
        recent_history=tuple(state.history[-selected_policy.recent_history_turns:]),
        asked_attribute=asked_attribute,
        rule_intent=intent.intent,
        rule_intent_confidence=intent.confidence,
        rule_evidence=intent.evidence,
        fallback_reasons=decision.reasons,
    )


def merge_rule_and_semantic(
    rule_update: StateUpdate,
    semantic_update: StateUpdate,
    *,
    prefer_semantic_intent: bool,
) -> StateUpdate:
    """Fill rule gaps with semantic output while preserving deterministic facts."""

    merged = StateUpdate(
        intent=(semantic_update.intent if prefer_semantic_intent and semantic_update.intent else rule_update.intent),
        hard_constraint={**semantic_update.hard_constraint, **rule_update.hard_constraint},
        soft_constraint={**semantic_update.soft_constraint, **rule_update.soft_constraint},
        no_preference=set(semantic_update.no_preference) | set(rule_update.no_preference),
        rejected_values={**semantic_update.rejected_values, **rule_update.rejected_values},
        boundary_attributes=set(semantic_update.boundary_attributes) | set(rule_update.boundary_attributes),
        override=rule_update.override or semantic_update.override,
        clear_hard_constraint=rule_update.clear_hard_constraint or semantic_update.clear_hard_constraint,
        clear_soft_constraint=rule_update.clear_soft_constraint or semantic_update.clear_soft_constraint,
    )
    return merged


class CallableSemanticResolver:
    """Adapter for an LLM client that returns a strict JSON-like mapping.

    The callable owns provider configuration, authentication, prompting, and
    retries. This adapter only validates and normalizes its structured result.
    """

    def __init__(self, call: Callable[[SemanticRequest], Mapping[str, Any] | None]) -> None:
        self.call = call

    def resolve(self, request: SemanticRequest) -> SemanticResolution | None:
        payload = self.call(request)
        if not isinstance(payload, Mapping):
            return None
        raw_intent = str(payload.get("intent") or "").lower()
        intent = raw_intent if raw_intent in {"buying", "browsing"} else None
        update = StateUpdate.from_raw(
            intent=intent,
            hard_constraint=_mapping(payload.get("hard_constraint")),
            soft_constraint=_mapping(payload.get("soft_constraint")),
            no_preference=_names(payload.get("no_preference")),
            rejected_values=_mapping(payload.get("rejected_values")),
            override=bool(payload.get("override", False)),
            clear_hard_constraint=bool(payload.get("clear_hard_constraint", False)),
            clear_soft_constraint=bool(payload.get("clear_soft_constraint", False)),
        )
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return SemanticResolution(update=update, confidence=max(0.0, min(1.0, confidence)))


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _names(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(item) for item in value}
    return set()
