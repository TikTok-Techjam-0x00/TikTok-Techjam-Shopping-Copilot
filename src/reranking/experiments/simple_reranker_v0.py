"""Behavioral replay of the first SimpleReranker scoring formula.

The original implementation was introduced in git commit ``799e8c1``. Commit
``66c0579`` adapted the same formula from Item inheritance to the current
``Candidate.item`` composition contract. This experiment keeps that historical
formula while adding only ``rank_all()`` for the offline Replay Evaluator.

It intentionally does not use the later ConstraintMatcher, CandidateSignals,
Feasibility Tier, Soft Penalty strategy, or RuleFuzzyScorer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ...item import Candidate, RankedCandidate
from ..reranker import (
    ShoppingStateInput,
    _constraint_map,
    _prepare_candidates,
    _shopping_intent,
    _state_constraints,
    _state_value,
)


ATTRIBUTE_ORDER = (
    "category",
    "use_case",
    "feature",
    "size",
    "material",
    "budget",
    "style",
    "color",
    "brand",
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(r"\d+(?:\.\d+)?")


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if _is_sequence(value):
        return " ".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _tokens(value: object) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(_text(value)) if len(token) > 1}


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        match = PRICE_RE.search(value.replace(",", ""))
        if match:
            return float(match.group())
    return None


def _detail_value(product: Mapping[str, Any], attribute: str) -> object:
    details = product.get("details")
    if not isinstance(details, Mapping):
        return None
    wanted = attribute.casefold().replace("_", " ")
    for key, value in details.items():
        if str(key).casefold().replace("_", " ") == wanted:
            return value
    return None


def _attribute_text(product: Mapping[str, Any], attribute: str) -> str:
    explicit = product.get(attribute)
    detail = _detail_value(product, attribute)
    if attribute == "category":
        explicit = product.get("categories") or explicit
    elif attribute == "brand":
        explicit = product.get("store") or _detail_value(product, "brand") or explicit
    elif attribute == "feature":
        explicit = product.get("features") or explicit

    common = {
        "title": product.get("title"),
        "categories": product.get("categories"),
        "features": product.get("features"),
        "description": product.get("description"),
    }
    return _text([explicit, detail, common])


def _match_ratio(requested: object, product_text: str) -> float:
    requested_tokens = _tokens(requested)
    if not requested_tokens:
        return 0.0
    product_tokens = _tokens(product_text)
    return len(requested_tokens & product_tokens) / len(requested_tokens)


def _budget_bounds(value: object) -> tuple[float | None, float | None]:
    if isinstance(value, Mapping):
        minimum = _numeric(value.get("min") if "min" in value else value.get("price_min"))
        maximum = _numeric(value.get("max") if "max" in value else value.get("price_max"))
        return minimum, maximum
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None, float(value)
    if isinstance(value, str):
        numbers = [float(item) for item in PRICE_RE.findall(value.replace(",", ""))]
        lowered = value.lower()
        if len(numbers) >= 2:
            return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
        if numbers and any(word in lowered for word in ("over", "above", "minimum", "at least")):
            return numbers[0], None
        if numbers:
            return None, numbers[0]
    return None, None


def _budget_match(product: Mapping[str, Any], requested: object) -> tuple[bool, str | None]:
    price = _numeric(product.get("price"))
    minimum, maximum = _budget_bounds(requested)
    if price is None or (minimum is None and maximum is None):
        return False, None
    if minimum is not None and price < minimum:
        return False, "budget:below_minimum"
    if maximum is not None and price > maximum:
        return False, "budget:above_maximum"
    return True, None


def _profile_match_ratio(product: Mapping[str, Any], user_profile: Mapping[str, Any]) -> float:
    tags = user_profile.get("preference_tags")
    if not _is_sequence(tags) or not tags:
        return 0.0
    searchable = _tokens(
        {
            "title": product.get("title"),
            "features": product.get("features"),
            "description": product.get("description"),
            "details": product.get("details"),
        }
    )
    normalized_tags = [str(tag).lower() for tag in tags if str(tag).strip()]
    if not normalized_tags:
        return 0.0
    matched = sum(1 for tag in normalized_tags if tag in searchable)
    return matched / len(normalized_tags)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    order = {attribute: index for index, attribute in enumerate(ATTRIBUTE_ORDER)}
    return sorted(set(values), key=lambda value: (order.get(value, len(order)), value))


def _score_candidate(
    candidate: Any,
    hard: Mapping[str, Any],
    soft: Mapping[str, Any],
    rejected: Mapping[str, Any],
    user_profile: Mapping[str, Any],
    shopping_intent: str,
) -> tuple[float, list[str], list[str]]:
    matched: list[str] = []
    violations: list[str] = []
    hard_match_total = 0.0
    soft_match_total = 0.0

    for attribute, requested in hard.items():
        if attribute == "budget":
            is_match, violation = _budget_match(candidate.product, requested)
            if is_match:
                matched.append(attribute)
                hard_match_total += 1.0
            elif violation:
                violations.append(violation)
            continue
        ratio = _match_ratio(requested, _attribute_text(candidate.product, attribute))
        if ratio >= 0.6:
            matched.append(attribute)
            hard_match_total += ratio
        else:
            violations.append(f"{attribute}:not_matched")

    for attribute, requested in soft.items():
        if attribute == "budget":
            is_match, _ = _budget_match(candidate.product, requested)
            ratio = 1.0 if is_match else 0.0
        else:
            ratio = _match_ratio(requested, _attribute_text(candidate.product, attribute))
        if ratio > 0:
            matched.append(attribute)
            soft_match_total += ratio

    for attribute, rejected_values in rejected.items():
        values = rejected_values if _is_sequence(rejected_values) else [rejected_values]
        product_text = _attribute_text(candidate.product, attribute)
        for rejected_value in values:
            if _match_ratio(rejected_value, product_text) >= 0.8:
                violations.append(f"{attribute}:rejected:{_text(rejected_value)[:40]}")

    hard_ratio = hard_match_total / len(hard) if hard else 0.0
    soft_ratio = soft_match_total / len(soft) if soft else 0.0
    profile_ratio = _profile_match_ratio(candidate.product, user_profile)
    violation_penalty = 1.0 if violations else 0.0

    if shopping_intent == "browsing":
        score = (
            0.70 * candidate.retrieval_score
            + 0.10 * hard_ratio
            + 0.10 * soft_ratio
            + 0.10 * profile_ratio
            - 0.75 * violation_penalty
        )
    else:
        score = (
            0.55 * candidate.retrieval_score
            + 0.25 * hard_ratio
            + 0.15 * soft_ratio
            + 0.05 * profile_ratio
            - 0.80 * violation_penalty
        )
    return score, _ordered_unique(matched), violations


class InitialSimpleReranker:
    """The original placeholder formula on the current shared object contract."""

    historical_source_commit = "799e8c1e88ee4d5632aad188452736d1255be54e"
    composition_adapter_commit = "66c0579b99a0a3c621ff184f6c553268e79486a8"

    def rerank(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> list[RankedCandidate]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        return self._rank(shopping_state, candidates_100, limit=top_k)

    def rank_all(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
    ) -> list[RankedCandidate]:
        return self._rank(shopping_state, candidates_100, limit=100)

    def _rank(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        *,
        limit: int,
    ) -> list[RankedCandidate]:
        prepared = _prepare_candidates(candidates_100)
        hard, soft = _state_constraints(shopping_state)
        shopping_intent = _shopping_intent(shopping_state)
        rejected = _constraint_map(_state_value(shopping_state, "rejected_values"))
        embedded_profile = _state_value(shopping_state, "user_profile")
        profile = embedded_profile if isinstance(embedded_profile, Mapping) else {}

        scored: list[tuple[float, Any, list[str], list[str]]] = []
        for candidate in prepared:
            score, matched, violations = _score_candidate(
                candidate,
                hard,
                soft,
                rejected,
                profile,
                shopping_intent,
            )
            scored.append((score, candidate, matched, violations))

        scored.sort(key=lambda row: (-row[0], row[1].original_index, row[1].parent_asin))
        result: list[RankedCandidate] = []
        for rank, (score, candidate, matched, violations) in enumerate(scored[:limit], start=1):
            result.append(
                RankedCandidate.from_candidate(
                    candidate.source,
                    rerank_rank=rank,
                    rerank_score=round(score, 6),
                    matched=matched,
                    violation=violations,
                )
            )
        return result


__all__ = ["InitialSimpleReranker"]
