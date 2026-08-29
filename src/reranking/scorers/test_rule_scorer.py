from __future__ import annotations

import dataclasses
import unittest

from src.attribute import AttributeName, normalize_attribute_map
from src.item import Item
from src.reranking.scorers import (
    FastRuleFuzzyScorer,
    RuleFuzzyScorer,
    RuleFuzzyScorerConfig,
)


class RuleFuzzyScorerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = RuleFuzzyScorer()
        self.relevant = Item.from_dict(
            {
                "parent_asin": "RELEVANT",
                "title": "Black Lightweight Mesh Running Shoes",
                "categories": ["Shoes", "Running"],
                "features": ["Lightweight", "Waterproof", "Comfortable"],
                "description": ["Cushioned outdoor training shoe"],
                "details": {"Material": "Mesh", "Color": "Black"},
                "price": 75,
                "store": "Trail Works",
            }
        )
        self.unrelated = Item.from_dict(
            {
                "parent_asin": "UNRELATED",
                "title": "Formal Red Leather Handbag",
                "categories": ["Handbags"],
                "features": ["Gold clasp"],
                "details": {"Material": "Leather", "Color": "Red"},
                "price": 160,
                "store": "City Bags",
            }
        )

    def test_relevant_product_scores_above_unrelated_product(self) -> None:
        hard = normalize_attribute_map(
            {"category": "running shoes", "material": "mesh", "budget": {"max": 100}}
        )
        soft = normalize_attribute_map(
            {"feature": ["lightweight", "waterproof"], "color": "black"}
        )
        relevant = self.scorer.score(
            self.relevant,
            hard_constraints=hard,
            soft_constraints=soft,
        )
        unrelated = self.scorer.score(
            self.unrelated,
            hard_constraints=hard,
            soft_constraints=soft,
        )
        self.assertGreater(relevant.score, unrelated.score)
        self.assertGreater(relevant.score, 0.7)
        self.assertIn("mesh", relevant.matched_terms)
        self.assertTrue(relevant.evidence)

    def test_category_hierarchy_is_compared_as_a_combined_view(self) -> None:
        result = self.scorer.score(
            self.relevant,
            hard_constraints=normalize_attribute_map({"category": "running shoes"}),
        )
        self.assertGreaterEqual(result.category_score, 0.7)
        self.assertEqual(
            result.attribute_scores[AttributeName.CATEGORY],
            result.category_score,
        )

    def test_phrase_aliases_match_water_resistant_to_waterproof(self) -> None:
        result = self.scorer.score(
            self.relevant,
            soft_constraints=normalize_attribute_map(
                {"feature": "water resistant"}
            ),
        )
        self.assertGreater(result.score, 0.7)
        self.assertIn("water resistant", result.matched_terms)

    def test_fuzzy_rescue_handles_a_typo_without_external_dependency(self) -> None:
        result = self.scorer.score(
            self.relevant,
            soft_constraints=normalize_attribute_map({"feature": "lightwieght"}),
        )
        self.assertGreater(result.fuzzy_score, 0.8)
        self.assertGreater(result.score, 0.6)

    def test_budget_score_rewards_nearby_values_and_handles_missing_price(self) -> None:
        budget = normalize_attribute_map({"budget": {"max": 100}})
        in_budget = self.scorer.score(self.relevant, hard_constraints=budget)
        over_budget = self.scorer.score(self.unrelated, hard_constraints=budget)
        missing = self.scorer.score(
            Item.from_dict({"parent_asin": "NO_PRICE"}),
            hard_constraints=budget,
        )
        self.assertEqual(in_budget.numeric_score, 1.0)
        self.assertGreater(in_budget.score, over_budget.score)
        self.assertEqual(missing.numeric_score, 0.0)

    def test_hard_constraints_have_more_weight_than_soft_constraints(self) -> None:
        mesh = normalize_attribute_map({"material": "mesh"})
        purple = normalize_attribute_map({"color": "purple"})
        hard_match = self.scorer.score(
            self.relevant,
            hard_constraints=mesh,
            soft_constraints=purple,
        )
        soft_match = self.scorer.score(
            self.relevant,
            hard_constraints=purple,
            soft_constraints=mesh,
        )
        self.assertGreater(hard_match.score, soft_match.score)

    def test_current_message_is_only_a_fallback_for_empty_state(self) -> None:
        fallback = self.scorer.score(
            self.relevant,
            query_text="lightweight running shoes",
        )
        structured = self.scorer.score(
            self.relevant,
            hard_constraints=normalize_attribute_map({"material": "mesh"}),
            query_text="red leather handbag",
        )
        self.assertGreater(fallback.score, 0.5)
        self.assertGreater(structured.score, 0.7)
        self.assertNotIn("message fallback", " ".join(structured.evidence))

    def test_empty_query_returns_zero_with_empty_diagnostics(self) -> None:
        result = self.scorer.score(self.relevant)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.attribute_scores, {})
        self.assertEqual(result.evidence, [])

    def test_configuration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            RuleFuzzyScorerConfig(phrase_weight=0.9)

    def test_s1_fast_is_diagnostic_and_score_equivalent(self) -> None:
        baseline = RuleFuzzyScorer()
        fast = FastRuleFuzzyScorer(product_cache_size=2, text_cache_size=128)
        hard = normalize_attribute_map(
            {"category": "running shoes", "material": "mesh", "budget": {"max": 100}}
        )
        soft = normalize_attribute_map(
            {"feature": ["lightwieght", "water resistant"], "color": "black"}
        )
        for product in (self.relevant, self.unrelated):
            expected = baseline.score(
                product,
                hard_constraints=hard,
                soft_constraints=soft,
                query_text="ignored when structured state exists",
            )
            actual = fast.score(
                product,
                hard_constraints=hard,
                soft_constraints=soft,
                query_text="ignored when structured state exists",
            )
            self.assertEqual(dataclasses.asdict(actual), dataclasses.asdict(expected))

        # A second pass must exercise both bounded caches without changing output.
        repeated = fast.score(
            self.relevant,
            hard_constraints=hard,
            soft_constraints=soft,
        )
        expected_repeated = baseline.score(
            self.relevant,
            hard_constraints=hard,
            soft_constraints=soft,
        )
        self.assertEqual(
            dataclasses.asdict(repeated),
            dataclasses.asdict(expected_repeated),
        )
        cache = fast.cache_info()
        self.assertEqual(cache["product_entries"], 2)
        self.assertGreater(cache["text"]["hits"], 0)

    def test_s1_fast_cache_sizes_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            FastRuleFuzzyScorer(product_cache_size=0)


if __name__ == "__main__":
    unittest.main()
