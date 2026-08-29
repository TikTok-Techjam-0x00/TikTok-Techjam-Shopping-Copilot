"""Optional semantic fallback for meanings that deterministic rules cannot resolve."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..attribute import AttributeName, AttributeValue
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
    semantic_confidence_threshold: float = 0.55
    intent_change_confidence_threshold: float = 0.85
    explicit_change_confidence_threshold: float = 0.70
    history_weight: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.recent_history_turns, bool)
            or not isinstance(self.recent_history_turns, int)
            or self.recent_history_turns < 0
        ):
            raise ValueError("recent_history_turns must be a non-negative integer")
        for name in (
            "intent_confidence_threshold",
            "semantic_confidence_threshold",
            "intent_change_confidence_threshold",
            "explicit_change_confidence_threshold",
            "history_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be between 0 and 1")


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


@dataclass(frozen=True, slots=True)
class FinalIntentResolution:
    """Validated per-turn intent after history-aware smoothing."""

    intent: Literal["buying", "browsing"]
    confidence: float
    source: Literal["rule", "llm", "history", "rule_fallback", "default"]
    smoothed: bool = False


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
    if intent.is_conflict:
        reasons.append("conflicting_rule_signals")
    elif intent.confidence < selected_policy.intent_confidence_threshold:
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
    if asked_attribute and has_attributes and not intent.is_conflict:
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
    recent_history = (
        tuple(state.history[-selected_policy.recent_history_turns:])
        if selected_policy.recent_history_turns
        else ()
    )
    current_state = state.to_dict()
    # History is carried separately with a hard turn limit. Avoid duplicating
    # an unbounded transcript or the prior raw message in the model payload.
    current_state.pop("history", None)
    current_state.pop("user_message", None)
    return SemanticRequest(
        message=message,
        current_state=current_state,
        recent_history=recent_history,
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
        # Semantic control flags are used only when the rule intent itself is
        # unresolved. A resolver may fill attribute gaps on a clear rule turn,
        # but it may not erase deterministic history or force an override.
        override=rule_update.override or (
            prefer_semantic_intent and semantic_update.override
        ),
        clear_hard_constraint=rule_update.clear_hard_constraint or (
            prefer_semantic_intent and semantic_update.clear_hard_constraint
        ),
        clear_soft_constraint=rule_update.clear_soft_constraint or (
            prefer_semantic_intent and semantic_update.clear_soft_constraint
        ),
    )
    return merged


def validate_semantic_resolution(
    resolution: object,
    *,
    policy: SemanticPolicy | None = None,
) -> tuple[SemanticResolution | None, tuple[str, ...]]:
    """Reject malformed or low-confidence resolver output before state mutation."""

    selected_policy = policy or SemanticPolicy()
    errors: list[str] = []
    if not isinstance(resolution, SemanticResolution):
        return None, ("invalid_resolution_type",)
    if not isinstance(resolution.update, StateUpdate):
        errors.append("invalid_update_type")
    if (
        isinstance(resolution.confidence, bool)
        or not isinstance(resolution.confidence, (int, float))
        or not math.isfinite(float(resolution.confidence))
        or not 0.0 <= float(resolution.confidence) <= 1.0
    ):
        errors.append("invalid_confidence")
    elif resolution.confidence < selected_policy.semantic_confidence_threshold:
        errors.append("semantic_confidence_below_threshold")

    update = resolution.update
    if isinstance(update, StateUpdate):
        if update.intent not in {None, "buying", "browsing"}:
            errors.append("invalid_intent")
        for field_name in ("hard_constraint", "soft_constraint", "rejected_values"):
            value = getattr(update, field_name)
            if not isinstance(value, dict) or any(
                not isinstance(name, AttributeName)
                or not isinstance(item, AttributeValue)
                or item.is_empty()
                for name, item in value.items()
            ):
                errors.append(f"invalid_{field_name}")
        if not isinstance(update.no_preference, set) or any(
            not isinstance(name, AttributeName) for name in update.no_preference
        ):
            errors.append("invalid_no_preference")
        if not isinstance(update.boundary_attributes, set) or any(
            not isinstance(name, AttributeName) for name in update.boundary_attributes
        ):
            errors.append("invalid_boundary_attributes")
        for field_name in ("override", "clear_hard_constraint", "clear_soft_constraint"):
            if not isinstance(getattr(update, field_name), bool):
                errors.append(f"invalid_{field_name}")
        if (
            update.override
            or update.clear_hard_constraint
            or update.clear_soft_constraint
        ) and resolution.confidence < selected_policy.explicit_change_confidence_threshold:
            errors.append("semantic_control_below_threshold")
        if not _has_meaningful_update(update):
            errors.append("empty_semantic_update")

    if errors:
        return None, tuple(dict.fromkeys(errors))
    return resolution, ()


def resolve_final_intent(
    rule_intent: IntentResult,
    semantic_resolution: SemanticResolution | None,
    state: ShoppingState,
    *,
    policy: SemanticPolicy | None = None,
) -> FinalIntentResolution:
    """Apply rule priority, then damp ambiguous intent changes with history."""

    selected_policy = policy or SemanticPolicy()
    rule_is_clear = (
        rule_intent.intent != "unknown"
        and not rule_intent.is_conflict
        and rule_intent.confidence >= selected_policy.intent_confidence_threshold
    )
    if rule_is_clear:
        return FinalIntentResolution(
            intent=rule_intent.intent,
            confidence=rule_intent.confidence,
            source="rule",
        )

    candidate_intent = semantic_resolution.update.intent if semantic_resolution else None
    candidate_confidence = semantic_resolution.confidence if semantic_resolution else 0.0
    source: Literal["llm", "rule_fallback", "default"] = "llm"
    if candidate_intent is None:
        if rule_intent.intent != "unknown":
            candidate_intent = rule_intent.intent
            candidate_confidence = rule_intent.confidence
            source = "rule_fallback"
        else:
            candidate_intent = state.intent
            candidate_confidence = rule_intent.confidence
            source = "default"

    if not state.history:
        return FinalIntentResolution(
            intent=candidate_intent,
            confidence=_unit(candidate_confidence),
            source=source,
        )

    previous_confidence = _unit(state.intent_confidence)
    if candidate_intent == state.intent:
        confidence = candidate_confidence
        smoothed = previous_confidence > 0.0
        if smoothed:
            confidence = (
                selected_policy.history_weight * previous_confidence
                + (1.0 - selected_policy.history_weight) * candidate_confidence
            )
        return FinalIntentResolution(
            intent=state.intent,
            confidence=_unit(confidence),
            source=source,
            smoothed=smoothed,
        )

    explicit_change = rule_intent.is_override or bool(
        semantic_resolution and semantic_resolution.update.override
    )
    threshold = (
        selected_policy.explicit_change_confidence_threshold
        if explicit_change
        else selected_policy.intent_change_confidence_threshold
    )
    if candidate_confidence >= threshold:
        return FinalIntentResolution(
            intent=candidate_intent,
            confidence=_unit(candidate_confidence),
            source=source,
        )

    retained_confidence = (
        selected_policy.history_weight * candidate_confidence
        + (1.0 - selected_policy.history_weight) * previous_confidence
    )
    return FinalIntentResolution(
        intent=state.intent,
        confidence=_unit(retained_confidence),
        source="history",
        smoothed=True,
    )


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
        allowed_fields = {
            "intent",
            "hard_constraint",
            "soft_constraint",
            "no_preference",
            "rejected_values",
            "override",
            "clear_hard_constraint",
            "clear_soft_constraint",
            "confidence",
        }
        if any(str(name) not in allowed_fields for name in payload):
            return None

        raw_intent = payload.get("intent")
        if raw_intent is not None and raw_intent not in {"buying", "browsing"}:
            return None
        hard_constraint = _strict_mapping(payload.get("hard_constraint"))
        soft_constraint = _strict_mapping(payload.get("soft_constraint"))
        rejected_values = _strict_mapping(payload.get("rejected_values"))
        no_preference = _strict_names(payload.get("no_preference"))
        if any(
            value is _INVALID
            for value in (
                hard_constraint,
                soft_constraint,
                rejected_values,
                no_preference,
            )
        ):
            return None
        control_fields = ("override", "clear_hard_constraint", "clear_soft_constraint")
        if any(name in payload and not isinstance(payload[name], bool) for name in control_fields):
            return None
        raw_confidence = payload.get("confidence")
        if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
            return None
        confidence = float(raw_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return None

        update = StateUpdate.from_raw(
            intent=raw_intent,
            hard_constraint=hard_constraint,
            soft_constraint=soft_constraint,
            no_preference=no_preference,
            rejected_values=rejected_values,
            boundary_attributes=no_preference,
            override=bool(payload.get("override", False)),
            clear_hard_constraint=bool(payload.get("clear_hard_constraint", False)),
            clear_soft_constraint=bool(payload.get("clear_soft_constraint", False)),
        )
        return SemanticResolution(update=update, confidence=confidence)


_INVALID = object()
_CANONICAL_ATTRIBUTE_NAMES = frozenset(name.value for name in AttributeName)


def _strict_mapping(value: object) -> Mapping[str, object] | None | object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return _INVALID
    result: dict[str, object] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip().lower()
        if name not in _CANONICAL_ATTRIBUTE_NAMES:
            return _INVALID
        result[name] = raw_value
    return result


def _strict_names(value: object) -> set[str] | object:
    if value is None:
        return set()
    if isinstance(value, str) or not isinstance(value, Sequence):
        return _INVALID
    result = {str(item).strip().lower() for item in value}
    return result if result <= _CANONICAL_ATTRIBUTE_NAMES else _INVALID


def _has_meaningful_update(update: StateUpdate) -> bool:
    return bool(
        update.intent
        or _has_attributes(update)
        or update.override
        or update.clear_hard_constraint
        or update.clear_soft_constraint
    )


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
