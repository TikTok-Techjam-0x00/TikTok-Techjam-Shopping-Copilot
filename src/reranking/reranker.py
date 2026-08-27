"""3A: deterministic reranking contract and a replaceable baseline scorer.

The module sits between retrieval and dialogue policy:

    shopping_state + candidates_100 -> candidates_10 -> 3B / Agent response

The current scorer is deliberately simple.  It gives the team a stable interface,
diagnostics, and tests while leaving one clear replacement point for a future
Cross-Encoder, LambdaMART model, or a better hand-tuned scoring function.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from ..item import Candidate, Candidates10, Item, RankedCandidate


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


class ShoppingStateProtocol(Protocol):
    """Structural interface expected from module 2's `shopping_state` class."""

    session_id: str
    user_profile: Mapping[str, Any]
    user_message: str
    turn: int
    intent: Literal["buying", "browsing"]
    hard_constraint: Mapping[str, Any]
    soft_constraint: Mapping[str, Any]
    no_prefernce: Any


ShoppingStateInput: TypeAlias = ShoppingStateProtocol | Mapping[str, Any]

@dataclass(frozen=True)
class _PreparedCandidate:
    original_index: int
    source: Candidate
    retrieval_score: float

    @property
    def parent_asin(self) -> str:
        return self.source.parent_asin

    @property
    def product(self) -> dict[str, Any]:
        return self.source.item.to_dict()


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


def _rank_fallback(index: int) -> float:
    """A stable score when module 1 has not supplied comparable retrieval scores."""
    return 1.0 / (1.0 + 0.15 * index)


def _prepare_candidates(
    retrieval_candidates: Sequence[Candidate | Mapping[str, Any]],
) -> list[_PreparedCandidate]:
    """Validate, deduplicate, and normalize retrieval scores to the 0..1 range.

    The input contract defines ``retrieval_score`` as higher-is-better.  If module 1
    only has a rank, it can omit the score and preserve its candidate ordering.
    """
    unique: list[tuple[int, Candidate, float | None]] = []
    seen: set[str] = set()
    for index, value in enumerate(retrieval_candidates[:100]):
        if not isinstance(value, Mapping):
            continue
        try:
            source = value if isinstance(value, Candidate) else Candidate.from_dict(value)
        except (TypeError, ValueError):
            continue
        if source.parent_asin in seen:
            continue
        seen.add(source.parent_asin)
        unique.append((index, source, source.retrieval_score))

    supplied = [score for _, _, score in unique if score is not None]
    minimum = min(supplied) if supplied else None
    maximum = max(supplied) if supplied else None

    prepared: list[_PreparedCandidate] = []
    for normalized_index, (original_index, source, raw_score) in enumerate(unique):
        if raw_score is not None and minimum is not None and maximum is not None and maximum > minimum:
            retrieval_score = (raw_score - minimum) / (maximum - minimum)
        else:
            retrieval_score = _rank_fallback(normalized_index)
        prepared.append(
            _PreparedCandidate(
                original_index=original_index,
                source=source,
                retrieval_score=retrieval_score,
            )
        )
    return prepared


def _constraint_map(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(attribute): constraint
        for attribute, constraint in value.items()
        if constraint not in (None, "", [], {}, ())
    }


def _state_value(shopping_state: ShoppingStateInput, field: str, default: Any = None) -> Any:
    if isinstance(shopping_state, Mapping):
        return shopping_state.get(field, default)
    return getattr(shopping_state, field, default)


def _no_preference_attributes(value: object) -> set[str]:
    """Normalize module 2's `no_prefernce` field to attribute names."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {str(attribute) for attribute in value}
    if _is_sequence(value):
        return {str(attribute) for attribute in value if str(attribute).strip()}
    return set()


def _state_constraints(
    shopping_state: ShoppingStateInput,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hard = _constraint_map(_state_value(shopping_state, "hard_constraint"))
    soft = _constraint_map(_state_value(shopping_state, "soft_constraint"))

    # `no_prefernce` means the user does not want this attribute to affect rank.
    # It is different from an explicit rejection such as "no leather".
    no_preference = _no_preference_attributes(
        _state_value(
            shopping_state,
            "no_prefernce",
            _state_value(shopping_state, "no_preference", []),
        )
    )
    for attribute in no_preference:
        hard.pop(attribute, None)
        soft.pop(attribute, None)
    return hard, soft


def _shopping_intent(shopping_state: ShoppingStateInput) -> str:
    """Read the module-2 mode: buying exploits; browsing explores."""
    value = str(_state_value(shopping_state, "intent", "")).strip().lower()
    if value not in {"buying", "browsing"}:
        raise ValueError("shopping_state.intent must be 'buying' or 'browsing'")
    return value


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
    candidate: _PreparedCandidate,
    hard: Mapping[str, Any],
    soft: Mapping[str, Any],
    rejected: Mapping[str, Any],
    user_profile: Mapping[str, Any],
    shopping_intent: str,
) -> tuple[float, list[str], list[str]]:
    """The placeholder scoring function to replace during ranking optimization."""
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
    # A stated rejection or hard-constraint mismatch should dominate a small
    # retrieval-score advantage. Keep violating candidates for diagnostics, but
    # push them below feasible candidates. A learned ranker can replace this
    # deliberately simple binary penalty later.
    violation_penalty = 1.0 if violations else 0.0

    if shopping_intent == "browsing":
        # Browsing keeps retrieval variety and soft/profile signals relatively
        # important. Cross-candidate diversity will replace this placeholder.
        score = (
            0.70 * candidate.retrieval_score
            + 0.10 * hard_ratio
            + 0.10 * soft_ratio
            + 0.10 * profile_ratio
            - 0.75 * violation_penalty
        )
    else:
        # Buying favors candidates that satisfy concrete requirements.
        score = (
            0.55 * candidate.retrieval_score
            + 0.25 * hard_ratio
            + 0.15 * soft_ratio
            + 0.05 * profile_ratio
            - 0.80 * violation_penalty
        )
    return score, _ordered_unique(matched), violations


class SimpleReranker:
    """Stable module-3A interface with a deliberately replaceable scorer."""

    def rerank(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> Candidates10:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        prepared = _prepare_candidates(candidates_100)
        hard, soft = _state_constraints(shopping_state)
        shopping_intent = _shopping_intent(shopping_state)
        rejected = _constraint_map(_state_value(shopping_state, "rejected_values"))
        embedded_profile = _state_value(shopping_state, "user_profile")
        profile = embedded_profile if isinstance(embedded_profile, Mapping) else {}

        scored: list[tuple[float, _PreparedCandidate, list[str], list[str]]] = []
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
        candidates_10: list[RankedCandidate] = []
        for rank, (score, candidate, matched, violations) in enumerate(scored[:top_k], start=1):
            candidates_10.append(
                RankedCandidate.from_candidate(
                    candidate.source,
                    rerank_rank=rank,
                    rerank_score=round(score, 6),
                    matched=matched,
                    violation=violations,
                )
            )
        return candidates_10


def rerank(
    shopping_state: ShoppingStateInput,
    candidates_100: Sequence[Candidate | Mapping[str, Any]],
    top_k: int = 10,
) -> Candidates10:
    """Convenience function for callers that do not need a component instance."""
    return SimpleReranker().rerank(shopping_state, candidates_100, top_k)


def recommendations_from_ranking(
    candidates_10: Sequence[RankedCandidate | Mapping[str, Any]], top_k: int = 10
) -> list[dict[str, str]]:
    """Convert `candidates_10` to the official recommendation shape."""
    recommendations: list[dict[str, str]] = []
    seen: set[str] = set()
    for ranked in candidates_10:
        if isinstance(ranked, RankedCandidate):
            parent_asin = ranked.item.parent_asin
        elif isinstance(ranked, Mapping):
            nested = ranked.get("item")
            nested_id = nested.get("parent_asin") if isinstance(nested, Mapping) else None
            parent_asin = str(ranked.get("parent_asin") or nested_id or "").strip()
        else:
            continue
        if not parent_asin or parent_asin in seen:
            continue
        seen.add(parent_asin)
        recommendations.append({"parent_asin": parent_asin})
        if len(recommendations) >= top_k:
            break
    return recommendations
