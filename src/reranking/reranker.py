"""3A: deterministic reranking contract and a replaceable baseline scorer.

The module sits between retrieval and dialogue policy:

    shopping_state + candidates_100 -> candidates_10 -> 3B / Agent response

The current scorer is deliberately simple.  It gives the team a stable interface,
diagnostics, and tests while leaving one clear replacement point for a future
Cross-Encoder, LambdaMART model, or a better hand-tuned scoring function.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, TypeAlias

from ..attribute import (
    AttributeMap,
    AttributeName,
    AttributeValue,
)
from ..item import Candidate, Candidates10, Item, RankedCandidate
from .constraint_matcher import (
    ConstraintMatch,
    ConstraintMatcher,
    MatchStatus,
)
from .feature_extractor import CandidateFeatureExtractor, CandidateSignals
from .scorers import RelevanceScorer, RuleFuzzyScorer


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


class HardConstraintStrategy(str, Enum):
    """How hard constraints affect final candidate ordering."""

    SOFT_PENALTY = "soft_penalty"
    FEASIBILITY_TIER = "feasibility_tier"


@dataclass(frozen=True, slots=True)
class RerankerStrategyConfig:
    """First-version intent routing and within-tier relevance weights."""

    browsing_strategy: HardConstraintStrategy = HardConstraintStrategy.SOFT_PENALTY
    buying_strategy: HardConstraintStrategy = HardConstraintStrategy.FEASIBILITY_TIER

    browsing_retrieval_weight: float = 0.60
    browsing_semantic_weight: float = 0.30
    browsing_profile_weight: float = 0.10

    buying_retrieval_weight: float = 0.20
    buying_semantic_weight: float = 0.35
    buying_hard_match_weight: float = 0.30
    buying_soft_match_weight: float = 0.10
    buying_profile_weight: float = 0.05

    def strategy_for(self, intent: str) -> HardConstraintStrategy:
        if intent == "browsing":
            return self.browsing_strategy
        if intent == "buying":
            return self.buying_strategy
        raise ValueError("intent must be 'buying' or 'browsing'")


class ShoppingStateProtocol(Protocol):
    """Structural interface expected from module 2's `shopping_state` class."""

    session_id: str
    user_profile: Mapping[str, Any]
    user_message: str
    turn: int
    intent: Literal["buying", "browsing"]
    hard_constraint: AttributeMap
    soft_constraint: AttributeMap
    no_prefernce: Sequence[AttributeName]


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
    def product(self) -> Item:
        return self.source.item


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _text(value: object) -> str:
    if isinstance(value, AttributeValue):
        return _text([value.values, value.details])
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if _is_sequence(value):
        return " ".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _tokens(value: object) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(_text(value)) if len(token) > 1}


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


def _hard_violation_label(match: ConstraintMatch) -> str:
    if match.attribute is AttributeName.BUDGET:
        evidence = " ".join(match.evidence)
        if "below minimum" in evidence:
            return "budget:below_minimum"
        if "above maximum" in evidence:
            return "budget:above_maximum"
    return f"{match.attribute.value}:not_matched"


def _rejected_violation_label(match: ConstraintMatch) -> str:
    requested = " ".join(match.requested_values)[:40]
    return f"{match.attribute.value}:rejected:{requested}"


def _matched_and_violations(
    signals: CandidateSignals,
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    violations: list[str] = []
    for match in signals.constraint_matches.hard:
        if match.status is MatchStatus.SATISFIED:
            matched.append(match.attribute.value)
        elif match.status is MatchStatus.VIOLATED:
            violations.append(_hard_violation_label(match))

    for match in signals.constraint_matches.soft:
        if match.status is MatchStatus.SATISFIED:
            matched.append(match.attribute.value)

    for match in signals.constraint_matches.rejected:
        if match.status is MatchStatus.VIOLATED:
            violations.append(_rejected_violation_label(match))
    return _ordered_unique(matched), violations


def _score_signals(
    signals: CandidateSignals,
    strategy: HardConstraintStrategy,
    config: RerankerStrategyConfig,
) -> float:
    semantic_score = signals.semantic_score or 0.0
    if strategy is HardConstraintStrategy.SOFT_PENALTY:
        return (
            config.browsing_retrieval_weight
            * signals.normalized_retrieval_score
            + config.browsing_semantic_weight * semantic_score
            + config.browsing_profile_weight * signals.profile_match_score
            + signals.soft_penalty_adjustment
        )
    return (
        config.buying_retrieval_weight * signals.normalized_retrieval_score
        + config.buying_semantic_weight * semantic_score
        + config.buying_hard_match_weight * signals.hard_match_score
        + config.buying_soft_match_weight * signals.soft_match_score
        + config.buying_profile_weight * signals.profile_match_score
    )


class SimpleReranker:
    """Stable module-3A interface with a deliberately replaceable scorer."""

    def __init__(
        self,
        *,
        constraint_matcher: ConstraintMatcher | None = None,
        feature_extractor: CandidateFeatureExtractor | None = None,
        relevance_scorer: RelevanceScorer | None = None,
        strategy_config: RerankerStrategyConfig | None = None,
    ) -> None:
        self.constraint_matcher = constraint_matcher or ConstraintMatcher()
        self.feature_extractor = feature_extractor or CandidateFeatureExtractor()
        self.relevance_scorer = relevance_scorer or RuleFuzzyScorer()
        self.strategy_config = strategy_config or RerankerStrategyConfig()

    def rerank(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> Candidates10:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")

        return self._rank(shopping_state, candidates_100, limit=top_k)

    def rank_all(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
    ) -> list[RankedCandidate]:
        """Return the complete order for offline diagnostics and replay evaluation.

        The production contract remains :meth:`rerank`, which returns at most ten
        candidates.  A complete order is needed offline to measure the target's
        exact promotion or demotion without exposing extra recommendations.
        """

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
        strategy = self.strategy_config.strategy_for(shopping_intent)
        rejected = _constraint_map(_state_value(shopping_state, "rejected_values"))
        embedded_profile = _state_value(shopping_state, "user_profile")
        profile = embedded_profile if isinstance(embedded_profile, Mapping) else {}
        query_text = str(_state_value(shopping_state, "user_message", "") or "")

        scored: list[
            tuple[float, _PreparedCandidate, CandidateSignals, list[str], list[str]]
        ] = []
        for candidate in prepared:
            constraint_matches = self.constraint_matcher.match_candidate(
                candidate.product,
                hard=hard,
                soft=soft,
                rejected=rejected,
            )
            relevance = self.relevance_scorer.score(
                candidate.product,
                hard_constraints=hard,
                soft_constraints=soft,
                query_text=query_text,
            )
            signals = self.feature_extractor.extract(
                candidate.source,
                constraint_matches,
                normalized_retrieval_score=candidate.retrieval_score,
                profile_match_score=_profile_match_ratio(candidate.product, profile),
                semantic_score=relevance.score,
            )
            score = _score_signals(signals, strategy, self.strategy_config)
            matched, violations = _matched_and_violations(signals)
            scored.append((score, candidate, signals, matched, violations))

        if strategy is HardConstraintStrategy.FEASIBILITY_TIER:
            scored.sort(
                key=lambda row: (
                    row[2].feasibility_tier,
                    -row[0],
                    row[1].original_index,
                    row[1].parent_asin,
                )
            )
        else:
            scored.sort(
                key=lambda row: (
                    -row[0],
                    row[1].original_index,
                    row[1].parent_asin,
                )
            )
        ranked_candidates: list[RankedCandidate] = []
        for rank, (score, candidate, _, matched, violations) in enumerate(
            scored[:limit], start=1
        ):
            ranked_candidates.append(
                RankedCandidate.from_candidate(
                    candidate.source,
                    rerank_rank=rank,
                    rerank_score=round(score, 6),
                    matched=matched,
                    violation=violations,
                )
            )
        return ranked_candidates


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
