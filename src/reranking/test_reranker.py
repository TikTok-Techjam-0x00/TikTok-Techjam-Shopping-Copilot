from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass
from typing import Any

from src.attribute import AttributeMap, AttributeName, normalize_attribute_map
from src.item import DATASET_FIELDS, Candidate, Item, RankedCandidate
from src.reranking import recommendations_from_ranking, rerank
from src.dialogue import decide_ask


@dataclass(frozen=True)
class MockShoppingState:
    """Temporary stand-in for module 2's class; not a production definition."""

    session_id: str
    user_profile: dict[str, Any]
    user_message: str
    turn: int
    intent: str
    hard_constraint: AttributeMap
    soft_constraint: AttributeMap
    no_prefernce: list[AttributeName]


PROFILE = {"preference_tags": ["comfort", "durability"]}

SHOPPING_STATE = MockShoppingState(
    session_id="test-session",
    user_profile=PROFILE,
    user_message="I want black lightweight running shoes under $100.",
    turn=2,
    intent="buying",
    hard_constraint=normalize_attribute_map({
        "category": "running shoes",
        "budget": {"max": 100},
        "material": "mesh",
    }),
    soft_constraint=normalize_attribute_map({
        "color": "black",
        "feature": ["lightweight", "comfortable"],
    }),
    no_prefernce=[AttributeName.BRAND],
)

CANDIDATES_100 = [
    Candidate.from_dict(
        {
            "parent_asin": "LEATHER",
            "bm25_score": 9.2,
            "dense_score": 0.82,
            "retrieval_score": 0.99,
            "retrieval_rank": 1,
            "product": {
                "parent_asin": "LEATHER",
                "title": "Black Leather Running Shoes",
                "categories": ["Shoes", "Running"],
                "features": ["Comfortable"],
                "details": {"Material": "Leather", "Color": "Black"},
                "price": 80,
            },
        }
    ),
    Candidate.from_dict(
        {
            "parent_asin": "MESH",
            "bm25_score": 8.7,
            "dense_score": 0.91,
            "retrieval_score": 0.90,
            "retrieval_rank": 2,
            "product": {
                "parent_asin": "MESH",
                "title": "Black Lightweight Mesh Running Shoes",
                "categories": ["Shoes", "Running"],
                "features": ["Lightweight", "Comfortable"],
                "details": {"Material": "Mesh", "Color": "Black"},
                "price": 75,
            },
        }
    ),
    Candidate.from_dict(
        {
            "parent_asin": "CASUAL",
            "retrieval_score": 0.70,
            "product": {
                "parent_asin": "CASUAL",
                "title": "White Casual Sneakers",
                "categories": ["Shoes", "Fashion Sneakers"],
                "features": ["Casual"],
                "details": {"Material": "Canvas", "Color": "White"},
                "price": 45,
            },
        }
    ),
]


class ProductModelTest(unittest.TestCase):
    def test_item_matches_official_catalog_fields(self) -> None:
        product = Item.from_dict(CANDIDATES_100[0].item.to_dict())
        self.assertEqual(set(product.to_dict()), set(DATASET_FIELDS))

    def test_pipeline_classes_compose_item(self) -> None:
        self.assertFalse(issubclass(Candidate, Item))
        self.assertFalse(issubclass(RankedCandidate, Item))
        self.assertTrue(all(isinstance(value, Candidate) for value in CANDIDATES_100))
        self.assertTrue(all(isinstance(value.item, Item) for value in CANDIDATES_100))

    def test_candidate_serialization_keeps_item_nested(self) -> None:
        serialized = CANDIDATES_100[0].to_dict()
        self.assertEqual(serialized["item"]["parent_asin"], "LEATHER")
        self.assertNotIn("title", serialized)
        self.assertEqual(serialized["bm25_score"], 9.2)


class SimpleRerankerTest(unittest.TestCase):
    def test_hard_constraints_can_change_retrieval_order(self) -> None:
        candidates_10 = rerank(SHOPPING_STATE, CANDIDATES_100)
        self.assertEqual(candidates_10[0].item.parent_asin, "MESH")
        leather = next(value for value in candidates_10 if value.item.parent_asin == "LEATHER")
        self.assertIn("material:not_matched", leather.violation)

    def test_output_is_reranked_candidate_list(self) -> None:
        candidates_10 = rerank(SHOPPING_STATE, CANDIDATES_100)
        self.assertTrue(all(isinstance(value, RankedCandidate) for value in candidates_10))
        self.assertEqual([value.rerank_rank for value in candidates_10], [1, 2, 3])
        self.assertNotIn("intent", candidates_10[0].matched)
        self.assertIn("category", candidates_10[0].matched)

    def test_output_preserves_item_and_retrieval_diagnostics(self) -> None:
        candidates_10 = rerank(SHOPPING_STATE, CANDIDATES_100)
        leather = next(value for value in candidates_10 if value.item.parent_asin == "LEATHER")
        self.assertIs(leather.item, CANDIDATES_100[0].item)
        self.assertEqual(leather.bm25_score, 9.2)
        self.assertEqual(leather.dense_score, 0.82)
        self.assertEqual(leather.retrieval_rank, 1)

    def test_no_prefernce_attribute_does_not_affect_ranking(self) -> None:
        shopping_state = MockShoppingState(
            session_id="browsing-session",
            user_profile=PROFILE,
            user_message="Show me some running shoes.",
            turn=1,
            intent="browsing",
            hard_constraint=normalize_attribute_map(
                {"brand": "a brand that does not exist"}
            ),
            soft_constraint={},
            no_prefernce=[AttributeName.BRAND],
        )
        candidates_10 = rerank(shopping_state, CANDIDATES_100)
        self.assertTrue(all("brand:not_matched" not in value.violation for value in candidates_10))

    def test_intent_only_accepts_buying_or_browsing(self) -> None:
        shopping_state = MockShoppingState(
            session_id="invalid-session",
            user_profile=PROFILE,
            user_message="Help me shop.",
            turn=1,
            intent="researching",
            hard_constraint={},
            soft_constraint={},
            no_prefernce=[],
        )
        with self.assertRaisesRegex(ValueError, "buying.*browsing"):
            rerank(shopping_state, CANDIDATES_100)

    def test_duplicate_and_invalid_candidates_are_removed(self) -> None:
        retrieval_output = [
            CANDIDATES_100[0],
            CANDIDATES_100[0],
            {"parent_asin": ""},
            CANDIDATES_100[1],
        ]
        candidates_10 = rerank(SHOPPING_STATE, retrieval_output)
        self.assertEqual(len(candidates_10), 2)

    def test_top_k_and_official_conversion(self) -> None:
        candidates_10 = rerank(SHOPPING_STATE, CANDIDATES_100, top_k=2)
        self.assertEqual(len(candidates_10), 2)
        recommendations = recommendations_from_ranking(candidates_10)
        self.assertEqual(
            recommendations,
            [{"parent_asin": value.item.parent_asin} for value in candidates_10],
        )

    def test_inputs_are_not_mutated(self) -> None:
        shopping_state = copy.deepcopy(SHOPPING_STATE)
        candidates_100 = copy.deepcopy(CANDIDATES_100)
        rerank(shopping_state, candidates_100)
        self.assertEqual(shopping_state, SHOPPING_STATE)
        self.assertEqual(candidates_100, CANDIDATES_100)

    def test_empty_candidates_return_empty_candidates_10(self) -> None:
        self.assertEqual(rerank(SHOPPING_STATE, []), [])

    def test_candidates_10_cannot_request_more_than_ten(self) -> None:
        with self.assertRaises(ValueError):
            rerank(SHOPPING_STATE, CANDIDATES_100, top_k=11)

    def test_candidates_10_are_accepted_by_3b(self) -> None:
        candidates_10 = rerank(SHOPPING_STATE, CANDIDATES_100)
        decision = decide_ask(
            {
                "turn": 2,
                "known_attributes": {"category": "running shoes"},
                "asked_attributes": [],
            },
            candidates_10,
        )
        self.assertIn("ask_attribute", decision)


if __name__ == "__main__":
    unittest.main()
