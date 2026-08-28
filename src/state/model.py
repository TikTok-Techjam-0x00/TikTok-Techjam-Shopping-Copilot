"""Canonical state objects shared with Retrieval, Reranking, and Dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..attribute import (
    AttributeMap,
    AttributeName,
    AttributeValue,
    attribute_map_to_dict,
    normalize_attribute_map,
    normalize_attribute_name,
)


Intent = Literal["buying", "browsing"]


def _copy_map(attributes: Mapping[AttributeName, AttributeValue]) -> AttributeMap:
    return {name: value.copy() for name, value in attributes.items()}


@dataclass(slots=True)
class StateUpdate:
    """Normalized changes extracted from one user message."""

    intent: Intent | None = None
    hard_constraint: AttributeMap = field(default_factory=dict)
    soft_constraint: AttributeMap = field(default_factory=dict)
    no_preference: set[AttributeName] = field(default_factory=set)
    rejected_values: AttributeMap = field(default_factory=dict)
    boundary_attributes: set[AttributeName] = field(default_factory=set)
    override: bool = False
    clear_hard_constraint: bool = False
    clear_soft_constraint: bool = False

    @classmethod
    def from_raw(
        cls,
        *,
        intent: Intent | None = None,
        hard_constraint: Mapping[str | AttributeName, object] | None = None,
        soft_constraint: Mapping[str | AttributeName, object] | None = None,
        no_preference: set[str | AttributeName] | None = None,
        rejected_values: Mapping[str | AttributeName, object] | None = None,
        boundary_attributes: set[str | AttributeName] | None = None,
        override: bool = False,
        clear_hard_constraint: bool = False,
        clear_soft_constraint: bool = False,
    ) -> StateUpdate:
        return cls(
            intent=intent,
            hard_constraint=normalize_attribute_map(hard_constraint),
            soft_constraint=normalize_attribute_map(soft_constraint),
            no_preference={normalize_attribute_name(value) for value in no_preference or set()},
            rejected_values=normalize_attribute_map(rejected_values),
            boundary_attributes={
                normalize_attribute_name(value)
                for value in boundary_attributes or set()
            },
            override=override,
            clear_hard_constraint=clear_hard_constraint,
            clear_soft_constraint=clear_soft_constraint,
        )


@dataclass(slots=True)
class ShoppingState:
    """One isolated conversation state implementing the shared team protocol."""

    session_id: str
    user_profile: dict[str, Any] = field(default_factory=dict)
    user_message: str = ""
    turn: int = 0
    intent: Intent = "browsing"
    hard_constraint: AttributeMap = field(default_factory=dict)
    soft_constraint: AttributeMap = field(default_factory=dict)
    no_prefernce: list[AttributeName] = field(default_factory=list)
    rejected_values: AttributeMap = field(default_factory=dict)
    asked_attributes: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    override_detected: bool = False
    intent_transitions: list[dict[str, Any]] = field(default_factory=list)
    boundary_detected: bool = False
    boundary_attributes: list[AttributeName] = field(default_factory=list)
    semantic_fallback_used: bool = False
    semantic_fallback_count: int = 0
    semantic_fallback_reasons: list[str] = field(default_factory=list)

    def _replace(self, target: AttributeMap, name: AttributeName, value: AttributeValue) -> None:
        target[name] = value.copy()
        if name in self.no_prefernce:
            self.no_prefernce.remove(name)

    def _merge(self, target: AttributeMap, source: Mapping[AttributeName, AttributeValue]) -> None:
        for name, value in source.items():
            target.setdefault(name, AttributeValue()).merge(value)
            if name in self.no_prefernce:
                self.no_prefernce.remove(name)

    def mark_attribute_asked(self, attribute: str | None) -> None:
        if attribute and attribute not in self.asked_attributes:
            self.asked_attributes.append(attribute)

    def apply_update(self, update: StateUpdate) -> None:
        """Apply one turn while preserving still-valid cross-category constraints."""

        self.override_detected = update.override
        self.boundary_detected = bool(update.boundary_attributes)
        if update.intent is not None:
            self.intent = update.intent

        if update.clear_hard_constraint:
            self.hard_constraint.clear()
            self.rejected_values.clear()

        new_category = update.hard_constraint.get(AttributeName.CATEGORY)
        if update.override and new_category is not None:
            self.hard_constraint.pop(AttributeName.CATEGORY, None)
            self.hard_constraint.pop(AttributeName.USE_CASE, None)
        if update.clear_soft_constraint:
            self.soft_constraint.clear()

        for name in update.no_preference:
            self.hard_constraint.pop(name, None)
            self.soft_constraint.pop(name, None)
            if name not in self.no_prefernce:
                self.no_prefernce.append(name)
        for name in update.boundary_attributes:
            if name not in self.boundary_attributes:
                self.boundary_attributes.append(name)

        # Explicit hard slots are single-valued and the newest turn wins.
        for name, value in update.hard_constraint.items():
            self._replace(self.hard_constraint, name, value)
        self._merge(self.soft_constraint, update.soft_constraint)
        for name, value in update.rejected_values.items():
            self._replace(self.rejected_values, name, value)
            positive = self.hard_constraint.get(name)
            if positive and set(positive.values) & set(value.values):
                self.hard_constraint.pop(name, None)

    @property
    def category(self) -> str | None:
        value = self.hard_constraint.get(AttributeName.CATEGORY)
        return value.values[0] if value and value.values else None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe pipeline/debug representation."""

        return {
            "session_id": self.session_id,
            "user_profile": dict(self.user_profile),
            "user_message": self.user_message,
            "turn": self.turn,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "hard_constraint": attribute_map_to_dict(self.hard_constraint),
            "soft_constraint": attribute_map_to_dict(self.soft_constraint),
            "no_prefernce": [name.value for name in self.no_prefernce],
            "rejected_values": attribute_map_to_dict(self.rejected_values),
            "asked_attributes": list(self.asked_attributes),
            "history": list(self.history),
            "override_detected": self.override_detected,
            "intent_transitions": [dict(value) for value in self.intent_transitions],
            "boundary_detected": self.boundary_detected,
            "boundary_attributes": [name.value for name in self.boundary_attributes],
            "semantic_fallback_used": self.semantic_fallback_used,
            "semantic_fallback_count": self.semantic_fallback_count,
            "semantic_fallback_reasons": list(self.semantic_fallback_reasons),
        }

    def copy(self) -> ShoppingState:
        return ShoppingState(
            session_id=self.session_id,
            user_profile=dict(self.user_profile),
            user_message=self.user_message,
            turn=self.turn,
            intent=self.intent,
            hard_constraint=_copy_map(self.hard_constraint),
            soft_constraint=_copy_map(self.soft_constraint),
            no_prefernce=list(self.no_prefernce),
            rejected_values=_copy_map(self.rejected_values),
            asked_attributes=list(self.asked_attributes),
            history=list(self.history),
            intent_confidence=self.intent_confidence,
            override_detected=self.override_detected,
            intent_transitions=[dict(value) for value in self.intent_transitions],
            boundary_detected=self.boundary_detected,
            boundary_attributes=list(self.boundary_attributes),
            semantic_fallback_used=self.semantic_fallback_used,
            semantic_fallback_count=self.semantic_fallback_count,
            semantic_fallback_reasons=list(self.semantic_fallback_reasons),
        )
