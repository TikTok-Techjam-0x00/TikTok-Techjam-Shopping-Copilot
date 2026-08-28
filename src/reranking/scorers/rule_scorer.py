"""S1: deterministic rule and fuzzy relevance baseline."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from ...attribute import (
    AttributeName,
    AttributeValue,
    normalize_attribute_name,
    product_attribute_values,
)
from ...item import Item
from .base import RelevanceScore


@dataclass(frozen=True, slots=True)
class RuleFuzzyScorerConfig:
    """Weights for the local S1 scorer; all outputs stay in the 0..1 range."""

    hard_constraint_weight: float = 2.0
    soft_constraint_weight: float = 1.0
    phrase_weight: float = 0.45
    token_overlap_weight: float = 0.35
    fuzzy_weight: float = 0.20
    fuzzy_rescue_threshold: float = 0.82
    global_text_weight: float = 0.15
    evidence_threshold: float = 0.65
    max_observations: int = 16
    max_fuzzy_characters: int = 256

    def __post_init__(self) -> None:
        if self.hard_constraint_weight <= 0 or self.soft_constraint_weight <= 0:
            raise ValueError("constraint weights must be positive")
        for name, value in (
            ("phrase_weight", self.phrase_weight),
            ("token_overlap_weight", self.token_overlap_weight),
            ("fuzzy_weight", self.fuzzy_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        component_total = self.phrase_weight + self.token_overlap_weight + self.fuzzy_weight
        if not math.isclose(component_total, 1.0, abs_tol=1e-9):
            raise ValueError("phrase/token/fuzzy weights must sum to 1")
        if not 0.0 <= self.global_text_weight <= 1.0:
            raise ValueError("global_text_weight must be between 0 and 1")
        if not 0.0 <= self.fuzzy_rescue_threshold <= 1.0:
            raise ValueError("fuzzy_rescue_threshold must be between 0 and 1")
        if not 0.0 <= self.evidence_threshold <= 1.0:
            raise ValueError("evidence_threshold must be between 0 and 1")
        if self.max_observations < 1 or self.max_fuzzy_characters < 16:
            raise ValueError("fuzzy comparison limits must be positive")


@dataclass(frozen=True, slots=True)
class _TextScore:
    combined: float
    phrase: float
    token: float
    fuzzy: float
    observed: str


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SPACE_RE = re.compile(r"\s+")

_PHRASE_ALIASES = (
    (re.compile(r"\bwater[\s-]?resistant\b"), "waterproof"),
    (re.compile(r"\bwater[\s-]?proof\b"), "waterproof"),
    (re.compile(r"\blight[\s-]?weight\b"), "lightweight"),
    (re.compile(r"\btee[\s-]?shirt\b"), "tshirt"),
    (re.compile(r"\bt[\s-]?shirt\b"), "tshirt"),
)

_TOKEN_ALIASES = {
    "jogging": "running",
    "jogger": "running",
    "trekking": "hiking",
    "trainer": "sneaker",
    "trainers": "sneaker",
    "sneakers": "sneaker",
    "comfortable": "comfort",
    "comfy": "comfort",
    "cushioned": "cushion",
    "grey": "gray",
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


def _normalize_text(value: object) -> str:
    text = _SPACE_RE.sub(" ", str(value).lower().replace("_", " ")).strip()
    for pattern, replacement in _PHRASE_ALIASES:
        text = pattern.sub(replacement, text)
    return text


def _token_key(token: str) -> str:
    token = _TOKEN_ALIASES.get(token, token)
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens_from_normalized(value: str) -> set[str]:
    return {_token_key(token) for token in _TOKEN_RE.findall(value)}


def _text_score(
    requested: str,
    observed_values: Sequence[str],
    config: RuleFuzzyScorerConfig,
) -> _TextScore:
    requested_text = _normalize_text(requested)
    if not requested_text:
        return _TextScore(0.0, 0.0, 0.0, 0.0, "")

    best = _TextScore(0.0, 0.0, 0.0, 0.0, "")
    requested_tokens = _tokens_from_normalized(requested_text)
    comparison_values = list(observed_values[: config.max_observations])
    combined_observed = " ".join(observed_values)
    if combined_observed:
        comparison_values.append(combined_observed)

    for observed in comparison_values:
        observed_text = _normalize_text(observed)
        if not observed_text:
            continue
        phrase_match = requested_text == observed_text or bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(requested_text)}(?![a-z0-9])",
                observed_text,
            )
        )
        phrase = float(phrase_match)
        observed_tokens = _tokens_from_normalized(observed_text)
        token = (
            len(requested_tokens & observed_tokens) / len(requested_tokens)
            if requested_tokens
            else 0.0
        )
        if phrase:
            candidate = _TextScore(1.0, 1.0, token, 1.0, observed)
            if candidate.combined > best.combined:
                best = candidate
            continue
        if token == 1.0:
            fuzzy = 1.0
        else:
            length_ratio = min(len(requested_text), len(observed_text)) / max(
                len(requested_text), len(observed_text)
            )
            should_compare_fuzzy = length_ratio >= 0.5 and (
                token > 0.0 or requested_text[0] == observed_text[0]
            )
            fuzzy = (
                SequenceMatcher(
                    None,
                    requested_text[: config.max_fuzzy_characters],
                    observed_text[: config.max_fuzzy_characters],
                ).ratio()
                if should_compare_fuzzy
                else 0.0
            )
        if len(requested_text) < 4 and not token:
            fuzzy = 0.0
        combined = (
            config.phrase_weight * phrase
            + config.token_overlap_weight * token
            + config.fuzzy_weight * fuzzy
        )
        if token == 1.0:
            combined = max(combined, 0.85)
        if (
            fuzzy >= config.fuzzy_rescue_threshold
            and min(len(requested_text), len(observed_text)) >= 5
        ):
            combined = max(combined, 0.85 * fuzzy)
        if combined > best.combined:
            best = _TextScore(combined, phrase, token, fuzzy, observed)
    return best


def _constraint_value(raw: object) -> AttributeValue:
    return raw.copy() if isinstance(raw, AttributeValue) else AttributeValue.from_raw(raw)


def _constraint_text_values(value: AttributeValue) -> list[str]:
    result = list(value.values)
    for detail_values in value.details.values():
        result.extend(detail_values)
    return _unique(result)


def _product_text(product: Item) -> list[str]:
    return _unique(
        _flatten_text(product.title)
        + _flatten_text(product.categories)
        + _flatten_text(product.features)
        + _flatten_text(product.description)
        + _flatten_text(product.details)
        + _flatten_text(product.store)
    )


def _attribute_observations(
    product: Item,
    attribute: AttributeName,
    full_product_text: Sequence[str],
) -> list[str]:
    structured = product_attribute_values(product.attributes, attribute)
    if attribute is AttributeName.CATEGORY:
        fallback = [*product.categories, product.title]
    elif attribute is AttributeName.BRAND:
        fallback = _flatten_text(product.store) + [product.title]
    elif attribute is AttributeName.FEATURE:
        fallback = [*product.features, product.title, *product.description]
    elif attribute in {AttributeName.USE_CASE, AttributeName.OTHER}:
        fallback = list(full_product_text)
    elif structured:
        fallback = []
    else:
        fallback = [product.title, *product.features, *_flatten_text(product.details)]
    return _unique([*structured, *fallback])


def _budget_score(product: Item, value: AttributeValue) -> tuple[float, str]:
    price = product.price
    if price is None:
        return 0.0, "product price is missing"

    minimum = value.minimum
    maximum = value.maximum
    if minimum is None and maximum is None and value.values:
        numbers = [
            float(number)
            for number in _NUMBER_RE.findall(" ".join(value.values).replace(",", ""))
        ]
        if len(numbers) >= 2:
            minimum, maximum = min(numbers[:2]), max(numbers[:2])
        elif numbers:
            maximum = numbers[0]
    if minimum is None and maximum is None:
        return 0.0, "budget has no numeric bound"
    if minimum is not None and price < minimum:
        distance = (minimum - price) / max(minimum, 1.0)
        return max(0.0, 1.0 - distance), f"price {price:g} below {minimum:g}"
    if maximum is not None and price > maximum:
        distance = (price - maximum) / max(maximum, 1.0)
        return max(0.0, 1.0 - distance), f"price {price:g} above {maximum:g}"
    return 1.0, f"price {price:g} within requested budget"


class RuleFuzzyScorer:
    """Score structured shopping needs against one product without a model."""

    def __init__(self, config: RuleFuzzyScorerConfig | None = None) -> None:
        self.config = config or RuleFuzzyScorerConfig()

    def score(
        self,
        product: Item,
        *,
        hard_constraints: Mapping[AttributeName | str, object] | None = None,
        soft_constraints: Mapping[AttributeName | str, object] | None = None,
        query_text: str = "",
    ) -> RelevanceScore:
        weighted_scores: list[tuple[float, float]] = []
        component_scores: list[tuple[_TextScore, float]] = []
        numeric_scores: list[tuple[float, float]] = []
        per_attribute: dict[AttributeName, list[tuple[float, float]]] = defaultdict(list)
        requested_texts: list[str] = []
        matched_terms: list[str] = []
        evidence: list[str] = []
        full_product_text = _product_text(product)
        observation_cache: dict[AttributeName, list[str]] = {}

        groups = (
            (hard_constraints, self.config.hard_constraint_weight, "hard"),
            (soft_constraints, self.config.soft_constraint_weight, "soft"),
        )
        for constraints, weight, group_name in groups:
            if not isinstance(constraints, Mapping):
                continue
            for raw_attribute, raw_value in constraints.items():
                attribute = normalize_attribute_name(raw_attribute)
                value = _constraint_value(raw_value)
                if attribute is AttributeName.BUDGET:
                    score, reason = _budget_score(product, value)
                    weighted_scores.append((score, weight))
                    numeric_scores.append((score, weight))
                    per_attribute[attribute].append((score, weight))
                    evidence.append(f"{group_name} budget: {reason} ({score:.2f})")
                    continue

                if attribute not in observation_cache:
                    observation_cache[attribute] = _attribute_observations(
                        product,
                        attribute,
                        full_product_text,
                    )
                observations = observation_cache[attribute]
                for requested in _constraint_text_values(value):
                    requested_texts.append(requested)
                    result = _text_score(requested, observations, self.config)
                    weighted_scores.append((result.combined, weight))
                    component_scores.append((result, weight))
                    per_attribute[attribute].append((result.combined, weight))
                    if result.combined >= self.config.evidence_threshold:
                        matched_terms.append(requested)
                        evidence.append(
                            f"{group_name} {attribute.value}: '{requested}' ~= "
                            f"'{result.observed}' ({result.combined:.2f})"
                        )

        # Current-message fallback is intentionally used only when State has no
        # structured positive requirement, preventing stale history leakage.
        if not weighted_scores and query_text.strip():
            result = _text_score(query_text, _product_text(product), self.config)
            weighted_scores.append((result.combined, 1.0))
            component_scores.append((result, 1.0))
            requested_texts.append(query_text)
            if result.combined >= self.config.evidence_threshold:
                matched_terms.append(query_text)
                evidence.append(
                    f"message fallback matched '{result.observed}' ({result.combined:.2f})"
                )

        total_weight = sum(weight for _, weight in weighted_scores)
        structured_score = (
            sum(score * weight for score, weight in weighted_scores) / total_weight
            if total_weight
            else 0.0
        )

        global_score = 0.0
        if requested_texts:
            global_result = _text_score(
                " ".join(requested_texts),
                full_product_text,
                self.config,
            )
            global_score = global_result.combined
        overall = structured_score
        if requested_texts:
            overall = (
                (1.0 - self.config.global_text_weight) * structured_score
                + self.config.global_text_weight * global_score
            )

        component_weight = sum(weight for _, weight in component_scores)
        phrase_score = (
            sum(result.phrase * weight for result, weight in component_scores)
            / component_weight
            if component_weight
            else 0.0
        )
        token_score = (
            sum(result.token * weight for result, weight in component_scores)
            / component_weight
            if component_weight
            else 0.0
        )
        fuzzy_score = (
            sum(result.fuzzy * weight for result, weight in component_scores)
            / component_weight
            if component_weight
            else 0.0
        )
        numeric_weight = sum(weight for _, weight in numeric_scores)
        numeric_score = (
            sum(score * weight for score, weight in numeric_scores) / numeric_weight
            if numeric_weight
            else 0.0
        )
        attribute_scores = {
            attribute: sum(score * weight for score, weight in scores)
            / sum(weight for _, weight in scores)
            for attribute, scores in per_attribute.items()
        }

        return RelevanceScore(
            score=overall,
            attribute_scores=attribute_scores,
            phrase_score=phrase_score,
            token_overlap_score=token_score,
            fuzzy_score=fuzzy_score,
            category_score=attribute_scores.get(AttributeName.CATEGORY, 0.0),
            numeric_score=numeric_score,
            matched_terms=_unique(matched_terms),
            evidence=evidence,
        )


_DEFAULT_SCORER = RuleFuzzyScorer()


def score_rule_relevance(
    product: Item,
    *,
    hard_constraints: Mapping[AttributeName | str, object] | None = None,
    soft_constraints: Mapping[AttributeName | str, object] | None = None,
    query_text: str = "",
) -> RelevanceScore:
    """Convenience wrapper using the default local S1 configuration."""
    return _DEFAULT_SCORER.score(
        product,
        hard_constraints=hard_constraints,
        soft_constraints=soft_constraints,
        query_text=query_text,
    )


__all__ = [
    "RuleFuzzyScorerConfig",
    "RuleFuzzyScorer",
    "score_rule_relevance",
]
