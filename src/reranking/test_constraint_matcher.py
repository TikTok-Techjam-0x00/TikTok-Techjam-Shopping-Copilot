from __future__ import annotations

import unittest

from src.attribute import AttributeName, normalize_attribute_map
from src.item import Item
from src.reranking.constraint_matcher import (
    ConstraintMatcher,
    ConstraintMatcherConfig,
    MatchStatus,
    MultiValuePolicy,
    match_constraint,
)


class ConstraintMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.product = Item.from_dict(
            {
                "parent_asin": "MESH",
                "title": "Black Lightweight Mesh Running Shoes",
                "categories": ["Shoes", "Running"],
                "features": ["Lightweight", "Comfortable"],
                "description": ["Waterproof upper for outdoor training"],
                "details": {
                    "Material": "Mesh",
                    "Color": "Black",
                    "Fit Type": "Regular Fit",
                    "Care Instructions": "Machine Wash",
                },
                "price": 75,
                "store": "Trail Works",
            }
        )

    def test_category_uses_the_combined_catalog_hierarchy(self) -> None:
        result = match_constraint(self.product, "category", "running shoes")
        self.assertIs(result.status, MatchStatus.SATISFIED)
        self.assertGreaterEqual(result.score, 0.9)
        self.assertEqual(result.requested_values, ["running shoes"])
        self.assertEqual(result.observed_values, ["Shoes", "Running"])
        self.assertTrue(result.evidence)

    def test_known_discrete_value_can_be_satisfied_or_violated(self) -> None:
        matched = match_constraint(self.product, AttributeName.MATERIAL, "mesh")
        mismatched = match_constraint(self.product, AttributeName.COLOR, "white")
        self.assertIs(matched.status, MatchStatus.SATISFIED)
        self.assertIs(mismatched.status, MatchStatus.VIOLATED)
        self.assertEqual(mismatched.score, 0.0)

    def test_missing_discrete_metadata_is_unknown_not_violated(self) -> None:
        product = Item.from_dict(
            {"parent_asin": "UNKNOWN", "title": "Everyday shoes"}
        )
        result = match_constraint(product, "material", "mesh")
        self.assertIs(result.status, MatchStatus.UNKNOWN)
        self.assertEqual(result.observed_values, [])

    def test_budget_distinguishes_satisfied_violated_and_unknown(self) -> None:
        satisfied = match_constraint(self.product, "budget", {"max": 100})
        above_maximum = match_constraint(self.product, "budget", {"max": 50})
        missing = match_constraint(
            Item.from_dict({"parent_asin": "NO_PRICE"}),
            "budget",
            {"max": 100},
        )
        self.assertIs(satisfied.status, MatchStatus.SATISFIED)
        self.assertIs(above_maximum.status, MatchStatus.VIOLATED)
        self.assertIn("above maximum", above_maximum.evidence[0])
        self.assertIs(missing.status, MatchStatus.UNKNOWN)

    def test_normalized_budget_attribute_value_is_supported(self) -> None:
        constraint = normalize_attribute_map({"budget": {"min": 50, "max": 80}})[
            AttributeName.BUDGET
        ]
        result = match_constraint(self.product, "budget", constraint)
        self.assertIs(result.status, MatchStatus.SATISFIED)
        self.assertEqual(result.requested_values, [">= 50", "<= 80"])

    def test_feature_all_policy_keeps_partial_match_unknown(self) -> None:
        result = match_constraint(
            self.product,
            "feature",
            ["lightweight", "insulated"],
        )
        self.assertIs(result.status, MatchStatus.UNKNOWN)
        self.assertAlmostEqual(result.score, 0.5)

    def test_feature_any_policy_is_an_experiment_switch(self) -> None:
        matcher = ConstraintMatcher(
            ConstraintMatcherConfig(feature_policy=MultiValuePolicy.ANY)
        )
        result = matcher.match(
            self.product,
            "feature",
            ["lightweight", "insulated"],
        )
        self.assertIs(result.status, MatchStatus.SATISFIED)
        self.assertEqual(result.score, 1.0)

    def test_open_attribute_can_use_product_text_fallback(self) -> None:
        result = match_constraint(self.product, "feature", "waterproof")
        self.assertIs(result.status, MatchStatus.SATISFIED)
        self.assertIn("product text", " ".join(result.evidence))

    def test_rejected_match_reverses_the_result(self) -> None:
        rejected_mesh = match_constraint(
            self.product,
            "material",
            "mesh",
            rejected=True,
        )
        rejected_leather = match_constraint(
            self.product,
            "material",
            "leather",
            rejected=True,
        )
        self.assertIs(rejected_mesh.status, MatchStatus.VIOLATED)
        self.assertEqual(rejected_mesh.score, 0.0)
        self.assertIs(rejected_leather.status, MatchStatus.SATISFIED)
        self.assertEqual(rejected_leather.score, 1.0)

    def test_rejected_value_with_missing_metadata_remains_unknown(self) -> None:
        product = Item.from_dict({"parent_asin": "UNKNOWN"})
        result = match_constraint(product, "material", "leather", rejected=True)
        self.assertIs(result.status, MatchStatus.UNKNOWN)

    def test_candidate_summary_counts_each_status_group(self) -> None:
        matcher = ConstraintMatcher()
        result = matcher.match_candidate(
            self.product,
            hard=normalize_attribute_map(
                {
                    "material": "mesh",
                    "color": "white",
                    "feature": "insulated",
                }
            ),
            soft=normalize_attribute_map({"brand": "Trail Works"}),
            rejected=normalize_attribute_map({"material": "mesh"}),
        )
        self.assertEqual(result.hard_satisfied_count, 1)
        self.assertEqual(result.hard_unknown_count, 1)
        self.assertEqual(result.hard_violation_count, 1)
        self.assertEqual(result.rejected_match_count, 1)
        self.assertIs(result.soft[0].status, MatchStatus.SATISFIED)

    def test_unknown_detail_is_available_through_singular_other(self) -> None:
        result = match_constraint(self.product, "other", "machine wash")
        self.assertIs(result.status, MatchStatus.SATISFIED)
        self.assertEqual(result.observed_values, ["Machine Wash"])

    def test_threshold_config_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            ConstraintMatcherConfig(lexical_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
