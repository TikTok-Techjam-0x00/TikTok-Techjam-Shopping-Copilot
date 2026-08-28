"""Attribute-aware constraint matching for Module 3A.

The matcher is deliberately deterministic and model-free. It separates a
confirmed mismatch from missing catalog evidence so later ranking strategies
can decide how strongly to penalize each case.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..attribute import (
    AttributeMap,
    AttributeName,
    AttributeValue,
    extract_product_attributes,
    normalize_attribute_name,
    product_attribute_values,
)
from ..item import Item


class MatchStatus(str, Enum):
    """Whether catalog evidence satisfies one user constraint."""

    SATISFIED = "satisfied"
    UNKNOWN = "unknown"
    VIOLATED = "violated"


class MultiValuePolicy(str, Enum):
    """How multiple requested values are interpreted."""

    ANY = "any"
    ALL = "all"


@dataclass(slots=True)
class ConstraintMatch:
    """Explainable result for one product attribute and one constraint."""

    attribute: AttributeName
    status: MatchStatus
    score: float
    requested_values: list[str] = field(default_factory=list)
    observed_values: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))


@dataclass(slots=True)
class CandidateConstraintMatches:
    """All matcher diagnostics for one candidate."""

    hard: list[ConstraintMatch] = field(default_factory=list)
    soft: list[ConstraintMatch] = field(default_factory=list)
    rejected: list[ConstraintMatch] = field(default_factory=list)

    @property
    def hard_satisfied_count(self) -> int:
        return sum(match.status is MatchStatus.SATISFIED for match in self.hard)

    @property
    def hard_unknown_count(self) -> int:
        return sum(match.status is MatchStatus.UNKNOWN for match in self.hard)

    @property
    def hard_violation_count(self) -> int:
        return sum(match.status is MatchStatus.VIOLATED for match in self.hard)

    @property
    def rejected_match_count(self) -> int:
        return sum(match.status is MatchStatus.VIOLATED for match in self.rejected)


@dataclass(frozen=True, slots=True)
class ConstraintMatcherConfig:
    """Experiment-friendly matcher choices with deterministic defaults."""

    lexical_threshold: float = 0.75
    fallback_threshold: float = 0.85
    feature_policy: MultiValuePolicy = MultiValuePolicy.ALL
    use_case_policy: MultiValuePolicy = MultiValuePolicy.ALL
    other_policy: MultiValuePolicy = MultiValuePolicy.ALL

    def __post_init__(self) -> None:
        for name, value in (
            ("lexical_threshold", self.lexical_threshold),
            ("fallback_threshold", self.fallback_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SPACE_RE = re.compile(r"\s+")

_CLOSED_WORLD_ATTRIBUTES = frozenset(
    {
        AttributeName.CATEGORY,
        AttributeName.MATERIAL,
        AttributeName.COLOR,
        AttributeName.SIZE,
        AttributeName.STYLE,
        AttributeName.BRAND,
    }
)

_VALUE_ALIASES = {
    "grey": "gray",
    "extra small": "xs",
    "x small": "xs",
    "small": "small",
    "s": "small",
    "medium": "medium",
    "m": "medium",
    "large": "large",
    "l": "large",
    "extra large": "xl",
    "x large": "xl",
    "water resistant": "waterproof",
}


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _flatten_text(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_flatten_text(nested))
        return result
    if _is_sequence(value):
        result = []
        for nested in value:
            result.extend(_flatten_text(nested))
        return result
    text = _SPACE_RE.sub(" ", str(value)).strip()
    return [text] if text else []


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalize_phrase(value: object) -> str:
    text = _SPACE_RE.sub(" ", str(value).strip().lower().replace("_", " "))
    return _VALUE_ALIASES.get(text, text)


def _token_key(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: object) -> set[str]:
    normalized = _normalize_phrase(value)
    return {_token_key(token) for token in _TOKEN_RE.findall(normalized)}


def _pair_score(requested: str, observed: str) -> float:
    requested_text = _normalize_phrase(requested)
    observed_text = _normalize_phrase(observed)
    if not requested_text or not observed_text:
        return 0.0
    if requested_text == observed_text:
        return 1.0
    if requested_text in observed_text:
        return 0.95

    requested_tokens = _tokens(requested_text)
    observed_tokens = _tokens(observed_text)
    if not requested_tokens or not observed_tokens:
        return 0.0
    intersection = requested_tokens & observed_tokens
    recall = len(intersection) / len(requested_tokens)
    union = requested_tokens | observed_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    if requested_tokens <= observed_tokens:
        return 0.90
    return 0.8 * recall + 0.2 * jaccard


def _attribute_value_values(value: AttributeValue | None) -> list[str]:
    if value is None:
        return []
    result = list(value.values)
    for detail_values in value.details.values():
        result.extend(detail_values)
    return _unique(result)


def _coerce_constraint(attribute: AttributeName, raw: object) -> AttributeValue:
    if isinstance(raw, AttributeValue):
        return raw.copy()
    if attribute is AttributeName.BUDGET:
        minimum, maximum, unit = _budget_bounds_from_raw(raw)
        if minimum is not None or maximum is not None:
            return AttributeValue.range(minimum=minimum, maximum=maximum, unit=unit)
    return AttributeValue.from_raw(raw)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    match = _NUMBER_RE.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def _budget_bounds_from_raw(
    raw: object,
) -> tuple[float | None, float | None, str | None]:
    if isinstance(raw, AttributeValue):
        minimum = raw.minimum
        maximum = raw.maximum
        unit = raw.unit
        if minimum is not None or maximum is not None:
            return minimum, maximum, unit
        raw = raw.values

    if isinstance(raw, Mapping):
        minimum = _number(raw.get("min", raw.get("minimum", raw.get("price_min"))))
        maximum = _number(raw.get("max", raw.get("maximum", raw.get("price_max"))))
        unit_value = raw.get("unit", raw.get("currency"))
        unit = str(unit_value) if unit_value not in (None, "") else None
        return minimum, maximum, unit

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None, _number(raw), None

    text = " ".join(_flatten_text(raw))
    numbers = [float(value) for value in _NUMBER_RE.findall(text.replace(",", ""))]
    lowered = text.lower()
    if len(numbers) >= 2:
        return min(numbers[0], numbers[1]), max(numbers[0], numbers[1]), None
    if not numbers:
        return None, None, None
    if any(word in lowered for word in ("over", "above", "minimum", "at least", ">=")):
        return numbers[0], None, None
    return None, numbers[0], None


def _product_mapping(product: Item | Mapping[str, Any]) -> Mapping[str, Any]:
    return product.to_dict() if isinstance(product, Item) else product


def _product_attributes(product: Item | Mapping[str, Any]) -> AttributeMap:
    if isinstance(product, Item):
        return product.attributes
    return extract_product_attributes(product)


def _fallback_parts(
    product: Item | Mapping[str, Any], attribute: AttributeName
) -> list[str]:
    raw = _product_mapping(product)
    details = raw.get("details")
    detail_values = _flatten_text(details) if isinstance(details, Mapping) else []

    if attribute is AttributeName.CATEGORY:
        return _flatten_text(raw.get("categories")) + _flatten_text(raw.get("title"))
    if attribute is AttributeName.BRAND:
        return _flatten_text(raw.get("store")) + _flatten_text(raw.get("title"))
    if attribute is AttributeName.FEATURE:
        return (
            _flatten_text(raw.get("features"))
            + _flatten_text(raw.get("title"))
            + _flatten_text(raw.get("description"))
            + detail_values
        )
    if attribute in {AttributeName.USE_CASE, AttributeName.OTHER}:
        return (
            _flatten_text(raw.get("title"))
            + _flatten_text(raw.get("categories"))
            + _flatten_text(raw.get("features"))
            + _flatten_text(raw.get("description"))
            + detail_values
        )
    return (
        _flatten_text(raw.get("title"))
        + _flatten_text(raw.get("features"))
        + detail_values
    )


def _best_scores(
    requested_values: Sequence[str], observed_values: Sequence[str]
) -> tuple[list[float], list[str | None]]:
    scores: list[float] = []
    best_observations: list[str | None] = []
    combined_observed = " ".join(observed_values)
    comparison_values = [*observed_values]
    if combined_observed:
        comparison_values.append(combined_observed)

    for requested in requested_values:
        best_score = 0.0
        best_observed: str | None = None
        for observed in comparison_values:
            score = _pair_score(requested, observed)
            if score > best_score:
                best_score = score
                best_observed = observed
        scores.append(best_score)
        best_observations.append(best_observed)
    return scores, best_observations


class ConstraintMatcher:
    """Match state constraints against one Item's shared attribute view."""

    def __init__(self, config: ConstraintMatcherConfig | None = None) -> None:
        self.config = config or ConstraintMatcherConfig()

    def policy_for(self, attribute: AttributeName) -> MultiValuePolicy:
        if attribute is AttributeName.FEATURE:
            return self.config.feature_policy
        if attribute is AttributeName.USE_CASE:
            return self.config.use_case_policy
        if attribute is AttributeName.OTHER:
            return self.config.other_policy
        return MultiValuePolicy.ANY

    def match(
        self,
        product: Item | Mapping[str, Any],
        attribute: AttributeName | str,
        requested: object,
        *,
        policy: MultiValuePolicy | None = None,
        rejected: bool = False,
    ) -> ConstraintMatch:
        """Match one constraint; ``rejected=True`` reverses its requirement."""
        canonical = normalize_attribute_name(attribute)
        if canonical is AttributeName.BUDGET:
            positive = self._match_budget(product, requested)
        else:
            positive = self._match_textual(
                product,
                canonical,
                requested,
                policy or self.policy_for(canonical),
            )
        return self._invert_rejected(positive) if rejected else positive

    def match_map(
        self,
        product: Item | Mapping[str, Any],
        constraints: Mapping[AttributeName | str, object] | None,
        *,
        rejected: bool = False,
    ) -> list[ConstraintMatch]:
        if not isinstance(constraints, Mapping):
            return []
        return [
            self.match(product, attribute, requested, rejected=rejected)
            for attribute, requested in constraints.items()
            if requested not in (None, "", [], {}, ())
        ]

    def match_candidate(
        self,
        product: Item | Mapping[str, Any],
        *,
        hard: Mapping[AttributeName | str, object] | None = None,
        soft: Mapping[AttributeName | str, object] | None = None,
        rejected: Mapping[AttributeName | str, object] | None = None,
    ) -> CandidateConstraintMatches:
        return CandidateConstraintMatches(
            hard=self.match_map(product, hard),
            soft=self.match_map(product, soft),
            rejected=self.match_map(product, rejected, rejected=True),
        )

    def _match_budget(
        self, product: Item | Mapping[str, Any], requested: object
    ) -> ConstraintMatch:
        minimum, maximum, unit = _budget_bounds_from_raw(requested)
        requested_values: list[str] = []
        if minimum is not None:
            requested_values.append(f">= {minimum:g}{f' {unit}' if unit else ''}")
        if maximum is not None:
            requested_values.append(f"<= {maximum:g}{f' {unit}' if unit else ''}")

        raw_product = _product_mapping(product)
        price = _number(raw_product.get("price"))
        observed_values = [f"{price:g} USD"] if price is not None else []
        if minimum is None and maximum is None:
            return ConstraintMatch(
                attribute=AttributeName.BUDGET,
                status=MatchStatus.UNKNOWN,
                score=0.0,
                requested_values=requested_values,
                observed_values=observed_values,
                evidence=["budget constraint has no parseable numeric bound"],
            )
        if price is None:
            return ConstraintMatch(
                attribute=AttributeName.BUDGET,
                status=MatchStatus.UNKNOWN,
                score=0.0,
                requested_values=requested_values,
                observed_values=[],
                evidence=["product price is missing"],
            )
        if minimum is not None and price < minimum:
            return ConstraintMatch(
                attribute=AttributeName.BUDGET,
                status=MatchStatus.VIOLATED,
                score=0.0,
                requested_values=requested_values,
                observed_values=observed_values,
                evidence=[f"price {price:g} is below minimum {minimum:g}"],
            )
        if maximum is not None and price > maximum:
            return ConstraintMatch(
                attribute=AttributeName.BUDGET,
                status=MatchStatus.VIOLATED,
                score=0.0,
                requested_values=requested_values,
                observed_values=observed_values,
                evidence=[f"price {price:g} is above maximum {maximum:g}"],
            )
        bounds = " and ".join(requested_values)
        return ConstraintMatch(
            attribute=AttributeName.BUDGET,
            status=MatchStatus.SATISFIED,
            score=1.0,
            requested_values=requested_values,
            observed_values=observed_values,
            evidence=[f"price {price:g} satisfies {bounds}"],
        )

    def _match_textual(
        self,
        product: Item | Mapping[str, Any],
        attribute: AttributeName,
        requested: object,
        policy: MultiValuePolicy,
    ) -> ConstraintMatch:
        constraint = _coerce_constraint(attribute, requested)
        requested_values = _attribute_value_values(constraint)
        attributes = _product_attributes(product)
        product_value = attributes.get(attribute)
        observed_values = product_attribute_values(attributes, attribute)
        if not requested_values:
            return ConstraintMatch(
                attribute=attribute,
                status=MatchStatus.UNKNOWN,
                score=0.0,
                requested_values=[],
                observed_values=observed_values,
                evidence=["constraint has no comparable text value"],
            )

        scores, best_observations = _best_scores(requested_values, observed_values)
        fallback_parts = _fallback_parts(product, attribute)
        fallback_scores, _ = _best_scores(requested_values, fallback_parts)
        evidence: list[str] = []
        effective_scores: list[float] = []
        for requested_value, structured_score, fallback_score, observed in zip(
            requested_values,
            scores,
            fallback_scores,
            best_observations,
        ):
            if structured_score >= self.config.lexical_threshold:
                effective_scores.append(structured_score)
                evidence.append(
                    f"requested '{requested_value}' matched observed "
                    f"'{observed}' ({structured_score:.2f})"
                )
            elif fallback_score >= self.config.fallback_threshold:
                effective_scores.append(fallback_score)
                evidence.append(
                    f"requested '{requested_value}' matched product text "
                    f"({fallback_score:.2f})"
                )
            else:
                effective_scores.append(structured_score)
                evidence.append(f"no evidence for requested '{requested_value}'")

        matched_flags = [
            score >= self.config.lexical_threshold for score in effective_scores
        ]
        if policy is MultiValuePolicy.ANY:
            satisfied = any(matched_flags)
            aggregate_score = max(effective_scores, default=0.0)
        else:
            satisfied = all(matched_flags)
            aggregate_score = (
                sum(effective_scores) / len(effective_scores)
                if effective_scores
                else 0.0
            )

        if satisfied:
            status = MatchStatus.SATISFIED
        elif product_value is not None and attribute in _CLOSED_WORLD_ATTRIBUTES:
            status = MatchStatus.VIOLATED
        else:
            status = MatchStatus.UNKNOWN

        return ConstraintMatch(
            attribute=attribute,
            status=status,
            score=aggregate_score,
            requested_values=requested_values,
            observed_values=observed_values,
            evidence=evidence,
        )

    @staticmethod
    def _invert_rejected(positive: ConstraintMatch) -> ConstraintMatch:
        if positive.status is MatchStatus.SATISFIED:
            status = MatchStatus.VIOLATED
            score = 0.0
            evidence = ["rejected value was observed", *positive.evidence]
        elif positive.status is MatchStatus.VIOLATED:
            status = MatchStatus.SATISFIED
            score = 1.0
            evidence = ["observed value does not match the rejection", *positive.evidence]
        else:
            status = MatchStatus.UNKNOWN
            score = 0.0
            evidence = ["insufficient evidence to evaluate rejection", *positive.evidence]
        return ConstraintMatch(
            attribute=positive.attribute,
            status=status,
            score=score,
            requested_values=list(positive.requested_values),
            observed_values=list(positive.observed_values),
            evidence=evidence,
        )


_DEFAULT_MATCHER = ConstraintMatcher()


def match_constraint(
    product: Item | Mapping[str, Any],
    attribute: AttributeName | str,
    requested: object,
    *,
    policy: MultiValuePolicy | None = None,
    rejected: bool = False,
) -> ConstraintMatch:
    """Convenience wrapper using the deterministic default configuration."""
    return _DEFAULT_MATCHER.match(
        product,
        attribute,
        requested,
        policy=policy,
        rejected=rejected,
    )


__all__ = [
    "MatchStatus",
    "MultiValuePolicy",
    "ConstraintMatch",
    "CandidateConstraintMatches",
    "ConstraintMatcherConfig",
    "ConstraintMatcher",
    "match_constraint",
]
