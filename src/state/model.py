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
ConstraintGroup = Literal["hard_constraint", "soft_constraint", "rejected_values"]


def _copy_map(attributes: Mapping[AttributeName, AttributeValue]) -> AttributeMap:
    return {name: value.copy() for name, value in attributes.items()}


@dataclass(slots=True)
class ConstraintProvenance:
    """Origin metadata for one active or superseded constraint."""

    attribute: AttributeName
    group: ConstraintGroup
    value: AttributeValue
    source_turn: int
    constraint_epoch: int
    confidence: float = 1.0

    def copy(self) -> ConstraintProvenance:
        return ConstraintProvenance(
            attribute=self.attribute,
            group=self.group,
            value=self.value.copy(),
            source_turn=self.source_turn,
            constraint_epoch=self.constraint_epoch,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value.to_dict(),
            "source_turn": self.source_turn,
            "constraint_epoch": self.constraint_epoch,
            "confidence": self.confidence,
        }


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
    confidence: float = 1.0

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
        confidence: float = 1.0,
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
            confidence=max(0.0, min(1.0, float(confidence))),
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
    semantic_validation_errors: list[str] = field(default_factory=list)
    intent_resolution_source: str = "default"
    intent_smoothed: bool = False
    constraint_epoch: int = 0
    last_override_turn: int | None = None
    constraint_provenance: dict[
        ConstraintGroup, dict[AttributeName, ConstraintProvenance]
    ] = field(default_factory=lambda: {
        "hard_constraint": {},
        "soft_constraint": {},
        "rejected_values": {},
    })
    constraint_history: list[ConstraintProvenance] = field(default_factory=list)
    asked_attributes_by_epoch: dict[int, list[str]] = field(default_factory=dict)

    def _archive(self, group: ConstraintGroup, name: AttributeName) -> None:
        record = self.constraint_provenance[group].pop(name, None)
        if record is not None:
            self.constraint_history.append(record.copy())

    def _clear_group(self, group: ConstraintGroup, target: AttributeMap) -> None:
        for name in list(target):
            self._archive(group, name)
        target.clear()

    def _record(
        self,
        group: ConstraintGroup,
        name: AttributeName,
        value: AttributeValue,
        *,
        source_turn: int,
        confidence: float,
    ) -> None:
        self._archive(group, name)
        self.constraint_provenance[group][name] = ConstraintProvenance(
            attribute=name,
            group=group,
            value=value.copy(),
            source_turn=source_turn,
            constraint_epoch=self.constraint_epoch,
            confidence=confidence,
        )

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
        self.asked_attributes_by_epoch[self.constraint_epoch] = list(
            self.asked_attributes
        )

    def apply_update(
        self,
        update: StateUpdate,
        *,
        source_turn: int | None = None,
    ) -> None:
        """Apply one turn while preserving still-valid cross-category constraints."""

        effective_turn = self.turn + 1 if source_turn is None else int(source_turn)

        self.override_detected = update.override
        self.boundary_detected = bool(update.boundary_attributes)
        if update.override:
            self.asked_attributes_by_epoch[self.constraint_epoch] = list(
                self.asked_attributes
            )
            self.constraint_epoch += 1
            self.last_override_turn = effective_turn
            self.asked_attributes = []
            self.asked_attributes_by_epoch.setdefault(self.constraint_epoch, [])
        if update.intent is not None:
            self.intent = update.intent

        if update.clear_hard_constraint:
            self._clear_group("hard_constraint", self.hard_constraint)
            self._clear_group("rejected_values", self.rejected_values)

        new_category = update.hard_constraint.get(AttributeName.CATEGORY)
        if update.override and new_category is not None:
            self._archive("hard_constraint", AttributeName.CATEGORY)
            self._archive("hard_constraint", AttributeName.USE_CASE)
            self.hard_constraint.pop(AttributeName.CATEGORY, None)
            self.hard_constraint.pop(AttributeName.USE_CASE, None)
        if update.clear_soft_constraint:
            self._clear_group("soft_constraint", self.soft_constraint)

        for name in update.no_preference:
            self._archive("hard_constraint", name)
            self._archive("soft_constraint", name)
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
            self._record(
                "hard_constraint",
                name,
                value,
                source_turn=effective_turn,
                confidence=update.confidence,
            )
        self._merge(self.soft_constraint, update.soft_constraint)
        for name, value in update.soft_constraint.items():
            self._record(
                "soft_constraint",
                name,
                self.soft_constraint[name],
                source_turn=effective_turn,
                confidence=update.confidence,
            )
        for name, value in update.rejected_values.items():
            self._replace(self.rejected_values, name, value)
            self._record(
                "rejected_values",
                name,
                value,
                source_turn=effective_turn,
                confidence=update.confidence,
            )
            positive = self.hard_constraint.get(name)
            if positive and set(positive.values) & set(value.values):
                self._archive("hard_constraint", name)
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
            "semantic_validation_errors": list(self.semantic_validation_errors),
            "intent_resolution_source": self.intent_resolution_source,
            "intent_smoothed": self.intent_smoothed,
            "constraint_epoch": self.constraint_epoch,
            "last_override_turn": self.last_override_turn,
            "constraint_provenance": {
                group: {
                    name.value: record.to_dict()
                    for name, record in records.items()
                }
                for group, records in self.constraint_provenance.items()
            },
            "constraint_history": [record.to_dict() | {
                "attribute": record.attribute.value,
                "group": record.group,
            } for record in self.constraint_history],
            "asked_attributes_by_epoch": {
                str(epoch): list(attributes)
                for epoch, attributes in self.asked_attributes_by_epoch.items()
            },
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
            semantic_validation_errors=list(self.semantic_validation_errors),
            intent_resolution_source=self.intent_resolution_source,
            intent_smoothed=self.intent_smoothed,
            constraint_epoch=self.constraint_epoch,
            last_override_turn=self.last_override_turn,
            constraint_provenance={
                group: {
                    name: record.copy() for name, record in records.items()
                }
                for group, records in self.constraint_provenance.items()
            },
            constraint_history=[record.copy() for record in self.constraint_history],
            asked_attributes_by_epoch={
                epoch: list(attributes)
                for epoch, attributes in self.asked_attributes_by_epoch.items()
            },
        )
