"""Shared scorer interface and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ...attribute import AttributeName
from ...item import Item


@dataclass(slots=True)
class RelevanceScore:
    """Normalized scorer output with explainable component signals."""

    score: float
    attribute_scores: dict[AttributeName, float] = field(default_factory=dict)
    phrase_score: float = 0.0
    token_overlap_score: float = 0.0
    fuzzy_score: float = 0.0
    category_score: float = 0.0
    numeric_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in (
            "score",
            "phrase_score",
            "token_overlap_score",
            "fuzzy_score",
            "category_score",
            "numeric_score",
        ):
            value = max(0.0, min(1.0, float(getattr(self, name))))
            setattr(self, name, value)
        self.attribute_scores = {
            attribute: max(0.0, min(1.0, float(value)))
            for attribute, value in self.attribute_scores.items()
        }


class RelevanceScorer(Protocol):
    """Interface shared by rule, Cross-Encoder, and future LTR scorers."""

    def score(
        self,
        product: Item,
        *,
        hard_constraints: Mapping[AttributeName | str, object] | None = None,
        soft_constraints: Mapping[AttributeName | str, object] | None = None,
        query_text: str = "",
    ) -> RelevanceScore:
        ...


__all__ = ["RelevanceScore", "RelevanceScorer"]
