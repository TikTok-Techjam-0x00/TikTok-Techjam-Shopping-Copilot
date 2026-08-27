from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateUpdate:
    """Structured changes extracted from one user message."""

    intent: str | None = None
    category: str | None = None
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: list[str] = field(default_factory=list)
    rejected_attributes: set[str] = field(default_factory=set)
    negative_constraints: dict[str, Any] = field(default_factory=dict)
    clear_soft_preferences: bool = False
    override: bool = False


@dataclass
class ShoppingState:
    """The current shopping intent and conversation memory for one session."""

    intent: str = "browsing"
    category: str | None = None
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: list[str] = field(default_factory=list)
    negative_constraints: dict[str, Any] = field(default_factory=dict)
    asked_attributes: set[str] = field(default_factory=set)
    rejected_attributes: set[str] = field(default_factory=set)
    history: list[str] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    intent_confidence: float = 0.0
    override_detected: bool = False

    def add_hard_constraint(self, attribute: str, value: Any) -> None:
        """Add or replace a required product attribute."""

        self.hard_constraints[attribute] = value
        self.rejected_attributes.discard(attribute)

    def add_soft_preference(self, value: str) -> None:
        """Add a preference without storing duplicates or blank values."""

        normalized = value.strip()
        if normalized and normalized not in self.soft_preferences:
            self.soft_preferences.append(normalized)

    def mark_attribute_asked(self, attribute: str) -> None:
        """Remember a clarification field so the agent does not repeat it."""

        self.asked_attributes.add(attribute)

    def reject_attribute(self, attribute: str) -> None:
        """Record that the user has no preference for an attribute."""

        self.rejected_attributes.add(attribute)
        self.hard_constraints.pop(attribute, None)

    def apply_update(self, update: StateUpdate) -> None:
        """Merge one parsed user turn into the current state."""

        self.override_detected = update.override

        if update.intent is not None:
            self.intent = update.intent

        # Category is single-valued, so a new category replaces the old one.
        if update.category is not None:
            if update.override and update.category != self.category:
                # Use-case words such as "running" are normally tied to the old
                # product category. Budget/size/color remain potentially valid.
                self.hard_constraints.pop("use_case", None)
            self.category = update.category.strip() or None

        # An explicit category override invalidates preferences tied to the old
        # product type. Generic hard constraints (for example a budget) remain.
        if update.clear_soft_preferences:
            self.soft_preferences.clear()

        for attribute in update.rejected_attributes:
            self.reject_attribute(attribute)

        for attribute, value in update.hard_constraints.items():
            self.add_hard_constraint(attribute, value)

        for attribute, value in update.negative_constraints.items():
            self.negative_constraints[attribute] = value
            if self.hard_constraints.get(attribute) == value:
                self.hard_constraints.pop(attribute, None)

        for preference in update.soft_preferences:
            self.add_soft_preference(preference)

    @property
    def known_attributes(self) -> dict[str, Any]:
        """Return the normalized slots consumed by retrieval and dialogue policy."""

        known = dict(self.hard_constraints)
        if self.category:
            known["category"] = self.category
        return known

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable pipeline contract."""

        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "category": self.category,
            "hard_constraints": dict(self.hard_constraints),
            "known_attributes": self.known_attributes,
            "soft_preferences": list(self.soft_preferences),
            "negative_constraints": dict(self.negative_constraints),
            "asked_attributes": sorted(self.asked_attributes),
            "rejected_attributes": sorted(self.rejected_attributes),
            "turn_count": self.turn_count,
            "turn": self.turn_count,
            "override_detected": self.override_detected,
            "user_profile": dict(self.user_profile),
        }
