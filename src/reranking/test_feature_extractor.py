from __future__ import annotations

import unittest

from src.attribute import AttributeName
from src.item import Candidate, Item
from src.reranking.constraint_matcher import (
    CandidateConstraintMatches,
    ConstraintMatch,
    MatchStatus,
)
from src.reranking.feature_extractor import (
    CandidateFeatureExtractor,
    ConstraintFeatureWeights,
)


def _match(
    attribute: AttributeName,
    status: MatchStatus,
    score: float,
) -> ConstraintMatch:
    return ConstraintMatch(
        attribute=attribute,
        status=status,
        score=score,
        requested_values=["requested"],
        observed_values=["observed"] if status is not MatchStatus.UNKNOWN else [],
        evidence=[status.value],
    )


class CandidateFeatureExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = Candidate(
            item=Item.from_dict({"parent_asin": "FEATURES"}),
            bm25_score=8.5,
            dense_score=0.75,
            retrieval_score=0.9,
            retrieval_rank=2,
        )
        self.extractor = CandidateFeatureExtractor()

    def test_counts_scores_and_retrieval_metadata_are_extracted(self) -> None:
        matches = CandidateConstraintMatches(
            hard=[
                _match(AttributeName.CATEGORY, MatchStatus.SATISFIED, 1.0),
                _match(AttributeName.FEATURE, MatchStatus.UNKNOWN, 0.5),
                _match(AttributeName.COLOR, MatchStatus.VIOLATED, 0.0),
            ],
            soft=[
                _match(AttributeName.STYLE, MatchStatus.SATISFIED, 0.9),
                _match(AttributeName.USE_CASE, MatchStatus.UNKNOWN, 0.25),
            ],
            rejected=[
                _match(AttributeName.MATERIAL, MatchStatus.VIOLATED, 0.0),
                _match(AttributeName.BRAND, MatchStatus.UNKNOWN, 0.0),
            ],
        )
        signals = self.extractor.extract(
            self.candidate,
            matches,
            normalized_retrieval_score=0.8,
            profile_match_score=0.6,
        )

        self.assertEqual(
            (
                signals.hard_satisfied_count,
                signals.hard_unknown_count,
                signals.hard_violation_count,
            ),
            (1, 1, 1),
        )
        self.assertEqual(
            (
                signals.soft_satisfied_count,
                signals.soft_unknown_count,
                signals.soft_violation_count,
            ),
            (1, 1, 0),
        )
        self.assertEqual(signals.rejected_match_count, 1)
        self.assertEqual(signals.rejected_unknown_count, 1)
        self.assertAlmostEqual(signals.hard_match_score, 0.5)
        self.assertAlmostEqual(signals.soft_match_score, 0.575)
        self.assertEqual(signals.bm25_score, 8.5)
        self.assertEqual(signals.dense_score, 0.75)
        self.assertEqual(signals.normalized_retrieval_score, 0.8)
        self.assertAlmostEqual(signals.retrieval_rank_score, 1 / 62)

    def test_default_soft_penalty_weights_match_the_plan(self) -> None:
        matches = CandidateConstraintMatches(
            hard=[
                _match(AttributeName.CATEGORY, MatchStatus.SATISFIED, 1.0),
                _match(AttributeName.COLOR, MatchStatus.VIOLATED, 0.0),
                _match(AttributeName.SIZE, MatchStatus.UNKNOWN, 0.0),
            ],
            soft=[_match(AttributeName.STYLE, MatchStatus.SATISFIED, 1.0)],
            rejected=[
                _match(AttributeName.MATERIAL, MatchStatus.VIOLATED, 0.0)
            ],
        )
        signals = self.extractor.extract(
            self.candidate,
            matches,
            normalized_retrieval_score=1.0,
        )
        # +2 hard satisfied -4 hard violation +1 soft satisfied -6 rejected.
        self.assertEqual(signals.hard_weighted_score, -2.0)
        self.assertEqual(signals.soft_weighted_score, 1.0)
        self.assertEqual(signals.rejected_weighted_score, -6.0)
        self.assertEqual(signals.soft_penalty_adjustment, -7.0)

    def test_feasibility_tiers_follow_strict_precedence(self) -> None:
        cases = [
            (CandidateConstraintMatches(), 0),
            (
                CandidateConstraintMatches(
                    hard=[_match(AttributeName.COLOR, MatchStatus.SATISFIED, 1.0)]
                ),
                0,
            ),
            (
                CandidateConstraintMatches(
                    hard=[_match(AttributeName.COLOR, MatchStatus.UNKNOWN, 0.0)]
                ),
                1,
            ),
            (
                CandidateConstraintMatches(
                    hard=[_match(AttributeName.COLOR, MatchStatus.VIOLATED, 0.0)]
                ),
                2,
            ),
            (
                CandidateConstraintMatches(
                    hard=[_match(AttributeName.COLOR, MatchStatus.VIOLATED, 0.0)],
                    rejected=[
                        _match(AttributeName.MATERIAL, MatchStatus.VIOLATED, 0.0)
                    ],
                ),
                3,
            ),
        ]
        for matches, expected_tier in cases:
            with self.subTest(tier=expected_tier):
                signals = self.extractor.extract(
                    self.candidate,
                    matches,
                    normalized_retrieval_score=0.5,
                )
                self.assertEqual(signals.feasibility_tier, expected_tier)

    def test_scores_are_clamped_and_non_finite_values_are_safe(self) -> None:
        signals = self.extractor.extract(
            self.candidate,
            CandidateConstraintMatches(),
            normalized_retrieval_score=2.0,
            profile_match_score=float("nan"),
            semantic_score=-1.0,
        )
        self.assertEqual(signals.normalized_retrieval_score, 1.0)
        self.assertEqual(signals.profile_match_score, 0.0)
        self.assertEqual(signals.semantic_score, 0.0)

    def test_weights_are_replaceable_without_rematching(self) -> None:
        extractor = CandidateFeatureExtractor(
            ConstraintFeatureWeights(hard_violated=-10.0)
        )
        matches = CandidateConstraintMatches(
            hard=[_match(AttributeName.COLOR, MatchStatus.VIOLATED, 0.0)]
        )
        signals = extractor.extract(
            self.candidate,
            matches,
            normalized_retrieval_score=0.5,
        )
        self.assertEqual(signals.hard_weighted_score, -10.0)
        self.assertIs(signals.constraint_matches, matches)


if __name__ == "__main__":
    unittest.main()
