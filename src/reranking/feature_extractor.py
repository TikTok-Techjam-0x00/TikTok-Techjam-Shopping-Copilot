"""Convert constraint matches into stable candidate-level ranking signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..item import Candidate
from .constraint_matcher import (
    CandidateConstraintMatches,
    ConstraintMatch,
    MatchStatus,
)


@dataclass(frozen=True, slots=True)
class ConstraintFeatureWeights:
    """Soft-penalty weights from the first rule-based experiment."""

    hard_satisfied: float = 2.0
    hard_unknown: float = 0.0
    hard_violated: float = -4.0
    soft_satisfied: float = 1.0
    soft_unknown: float = 0.0
    soft_violated: float = 0.0
    rejected_safe: float = 0.0
    rejected_unknown: float = 0.0
    rejected_matched: float = -6.0


@dataclass(slots=True)
class CandidateSignals:
    """All deterministic signals currently available for one candidate."""

    candidate: Candidate
    constraint_matches: CandidateConstraintMatches

    normalized_retrieval_score: float
    retrieval_rank_score: float
    bm25_score: float | None
    dense_score: float | None

    hard_satisfied_count: int
    hard_unknown_count: int
    hard_violation_count: int
    hard_match_score: float
    hard_weighted_score: float

    soft_satisfied_count: int
    soft_unknown_count: int
    soft_violation_count: int
    soft_match_score: float
    soft_weighted_score: float

    rejected_match_count: int
    rejected_unknown_count: int
    rejected_weighted_score: float

    profile_match_score: float
    semantic_score: float | None
    feasibility_tier: int
    soft_penalty_adjustment: float

    @property
    def parent_asin(self) -> str:
        return self.candidate.parent_asin


def _finite_score(value: float | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _unit_score(value: float | None) -> float:
    return max(0.0, min(1.0, _finite_score(value)))


def _mean_match_score(matches: list[ConstraintMatch]) -> float:
    if not matches:
        return 0.0
    return sum(match.score for match in matches) / len(matches)


def _status_count(matches: list[ConstraintMatch], status: MatchStatus) -> int:
    return sum(match.status is status for match in matches)


def _weighted_status_score(
    matches: list[ConstraintMatch],
    *,
    satisfied: float,
    unknown: float,
    violated: float,
) -> float:
    weights = {
        MatchStatus.SATISFIED: satisfied,
        MatchStatus.UNKNOWN: unknown,
        MatchStatus.VIOLATED: violated,
    }
    return sum(weights[match.status] for match in matches)


def _feasibility_tier(matches: CandidateConstraintMatches) -> int:
    if matches.rejected_match_count:
        return 3
    if matches.hard_violation_count:
        return 2
    if matches.hard_unknown_count:
        return 1
    return 0


class CandidateFeatureExtractor:
    """Create CandidateSignals without changing ranking order or candidates."""

    def __init__(self, weights: ConstraintFeatureWeights | None = None) -> None:
        self.weights = weights or ConstraintFeatureWeights()

    def extract(
        self,
        candidate: Candidate,
        constraint_matches: CandidateConstraintMatches,
        *,
        normalized_retrieval_score: float,
        profile_match_score: float = 0.0,
        semantic_score: float | None = None,
    ) -> CandidateSignals:
        hard = constraint_matches.hard
        soft = constraint_matches.soft
        rejected = constraint_matches.rejected

        hard_satisfied_count = _status_count(hard, MatchStatus.SATISFIED)
        hard_unknown_count = _status_count(hard, MatchStatus.UNKNOWN)
        hard_violation_count = _status_count(hard, MatchStatus.VIOLATED)
        soft_satisfied_count = _status_count(soft, MatchStatus.SATISFIED)
        soft_unknown_count = _status_count(soft, MatchStatus.UNKNOWN)
        soft_violation_count = _status_count(soft, MatchStatus.VIOLATED)
        rejected_match_count = _status_count(rejected, MatchStatus.VIOLATED)
        rejected_unknown_count = _status_count(rejected, MatchStatus.UNKNOWN)

        hard_weighted_score = _weighted_status_score(
            hard,
            satisfied=self.weights.hard_satisfied,
            unknown=self.weights.hard_unknown,
            violated=self.weights.hard_violated,
        )
        soft_weighted_score = _weighted_status_score(
            soft,
            satisfied=self.weights.soft_satisfied,
            unknown=self.weights.soft_unknown,
            violated=self.weights.soft_violated,
        )
        rejected_weighted_score = _weighted_status_score(
            rejected,
            satisfied=self.weights.rejected_safe,
            unknown=self.weights.rejected_unknown,
            violated=self.weights.rejected_matched,
        )

        rank = candidate.retrieval_rank
        retrieval_rank_score = 1.0 / (60.0 + rank) if rank and rank > 0 else 0.0
        soft_penalty_adjustment = (
            hard_weighted_score
            + soft_weighted_score
            + rejected_weighted_score
        )

        return CandidateSignals(
            candidate=candidate,
            constraint_matches=constraint_matches,
            normalized_retrieval_score=_unit_score(normalized_retrieval_score),
            retrieval_rank_score=retrieval_rank_score,
            bm25_score=candidate.bm25_score,
            dense_score=candidate.dense_score,
            hard_satisfied_count=hard_satisfied_count,
            hard_unknown_count=hard_unknown_count,
            hard_violation_count=hard_violation_count,
            hard_match_score=_mean_match_score(hard),
            hard_weighted_score=hard_weighted_score,
            soft_satisfied_count=soft_satisfied_count,
            soft_unknown_count=soft_unknown_count,
            soft_violation_count=soft_violation_count,
            soft_match_score=_mean_match_score(soft),
            soft_weighted_score=soft_weighted_score,
            rejected_match_count=rejected_match_count,
            rejected_unknown_count=rejected_unknown_count,
            rejected_weighted_score=rejected_weighted_score,
            profile_match_score=_unit_score(profile_match_score),
            semantic_score=(
                _unit_score(semantic_score) if semantic_score is not None else None
            ),
            feasibility_tier=_feasibility_tier(constraint_matches),
            soft_penalty_adjustment=soft_penalty_adjustment,
        )


_DEFAULT_EXTRACTOR = CandidateFeatureExtractor()


def extract_candidate_features(
    candidate: Candidate,
    constraint_matches: CandidateConstraintMatches,
    *,
    normalized_retrieval_score: float,
    profile_match_score: float = 0.0,
    semantic_score: float | None = None,
) -> CandidateSignals:
    """Convenience wrapper using the first-version feature weights."""
    return _DEFAULT_EXTRACTOR.extract(
        candidate,
        constraint_matches,
        normalized_retrieval_score=normalized_retrieval_score,
        profile_match_score=profile_match_score,
        semantic_score=semantic_score,
    )


__all__ = [
    "ConstraintFeatureWeights",
    "CandidateSignals",
    "CandidateFeatureExtractor",
    "extract_candidate_features",
]
