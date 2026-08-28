from __future__ import annotations

import unittest
from inspect import signature
from types import SimpleNamespace

from src.dialogue.three_b import (
    _candidate_diversity_signal,
    _product,
    _retrieval_items,
    _unavailable_attributes,
    _values,
    build_question,
    decide_ask,
    record_asked_attribute,
)
from src.item import Candidate, Item


def _candidate(**product: object) -> Candidate:
    return Candidate(item=Item.from_dict(product))


class AttributeValueStub:
    """不依赖具体值内部结构，模拟新版 AttributeValue。"""


CANDIDATES_100 = [
    _candidate(parent_asin="A", title="Wide black leather shoes", details={"Material": "Leather", "Color": "Black", "Size": "Wide"}),
    _candidate(parent_asin="B", title="Regular white canvas shoes", details={"Material": "Canvas", "Color": "White", "Size": "Regular"}),
    _candidate(parent_asin="C", title="Regular brown leather shoes", details={"Material": "Leather", "Color": "Brown", "Size": "Regular"}),
]


class AskAttributeSelectorTest(unittest.TestCase):
    def test_unknown_category_is_asked_first(self) -> None:
        decision = decide_ask({"turn": 1}, CANDIDATES_100)
        self.assertEqual(decision["ask_attribute"], "category")

    def test_never_repeats_known_asked_or_unavailable_attributes(self) -> None:
        state = {
            "turn": 3,
            "hard_constraint": {"category": "shoes", "use_case": "walking"},
            "asked_attributes": ["feature", "size"],
            "no_prefernce": ["material"],
        }
        decision = decide_ask(state, CANDIDATES_100)
        self.assertNotIn(
            decision["ask_attribute"],
            {"category", "use_case", "feature", "size", "material"},
        )

    def test_ranking_ambiguity_affects_selection(self) -> None:
        state = {
            "turn": 3,
            "hard_constraint": {"category": "shoes", "use_case": "walking"},
            "soft_constraint": {"feature": "comfortable"},
            "asked_attributes": ["size"],
        }
        decision = decide_ask(state, CANDIDATES_100)
        self.assertEqual(decision["ask_attribute"], "material")
        self.assertIn("leather", decision["message"])

    def test_user_profile_does_not_change_attribute_scoring(self) -> None:
        # Profile 已由 3A 用于排序；3B 不应再次把同一信号加到 attribute score。
        shared_state = {
            "turn": 3,
            "hard_constraint": {"category": "shoes"},
        }
        without_profile_signal = decide_ask(
            {**shared_state, "user_profile": {"preference_tags": []}},
            [],
        )
        with_profile_signal = decide_ask(
            {
                **shared_state,
                "user_profile": {
                    "preference_tags": ["material", "fit", "brand"],
                },
            },
            [],
        )
        self.assertEqual(without_profile_signal, with_profile_signal)
        self.assertEqual(with_profile_signal["ask_attribute"], "use_case")

    def test_diversity_maximum_is_24_for_low_and_high_cardinality(self) -> None:
        # 每个候选都包含全部值，使 coverage 和 normalized entropy 都精确为 1。
        def candidates_with_all_values(values: list[str]) -> list[Candidate]:
            return [
                _candidate(parent_asin=f"C{i}", details={"Material": values})
                for i in range(20)
            ]

        two_value_boost, _ = _candidate_diversity_signal(
            candidates_with_all_values(["value-1", "value-2"]),
            "material",
        )
        twenty_value_boost, _ = _candidate_diversity_signal(
            candidates_with_all_values([f"value-{i}" for i in range(20)]),
            "material",
        )
        self.assertAlmostEqual(two_value_boost, 24.0)
        self.assertAlmostEqual(twenty_value_boost, 24.0)

    def test_diversity_half_coverage_is_12(self) -> None:
        candidates = [
            _candidate(
                parent_asin=f"C{i}",
                details={"Material": ["cotton", "wool"]},
            )
            for i in range(10)
        ]
        candidates.extend(
            _candidate(parent_asin=f"M{i}", details={})
            for i in range(10, 20)
        )

        boost, _ = _candidate_diversity_signal(candidates, "material")
        self.assertAlmostEqual(boost, 12.0)

    def test_nearly_identical_candidate_values_have_near_zero_boost(self) -> None:
        candidates = [
            _candidate(parent_asin=f"C{i}", details={"Material": "Cotton"})
            for i in range(99)
        ]
        candidates.append(
            _candidate(parent_asin="P99", details={"Material": "Polyester"})
        )

        boost, _ = _candidate_diversity_signal(candidates, "material")
        self.assertGreater(boost, 0.0)
        self.assertLess(boost, 2.0)

    def test_turn_ten_does_not_ask_another_question(self) -> None:
        decision = decide_ask({"turn": 10}, CANDIDATES_100)
        self.assertEqual(decision, {"ask_attribute": None, "message": ""})

    def test_size_question_uses_ranking_options(self) -> None:
        question = build_question("size", ["small", "medium", "large"])
        self.assertIn("small, medium, or large", question)

    def test_caller_can_record_current_question_in_state(self) -> None:
        state = {"asked_attributes": ["category"]}
        record_asked_attribute(state, "size")
        self.assertEqual(state["asked_attributes"], ["category", "size"])

    def test_invalid_turn_falls_back_without_exception(self) -> None:
        decision = decide_ask({"turn": "second"}, CANDIDATES_100)
        self.assertEqual(decision["ask_attribute"], "category")

    def test_candidates_11_to_100_affect_diversity_calculation(self) -> None:
        candidates = [
            _candidate(parent_asin=f"C{i}", details={"Material": "Cotton"})
            for i in range(10)
        ]
        candidates.extend(
            _candidate(parent_asin=f"P{i}", details={"Material": "Polyester"})
            for i in range(10, 100)
        )
        candidates.extend(
            _candidate(parent_asin=f"S{i}", details={"Material": "Silk"})
            for i in range(100, 105)
        )

        items = _retrieval_items(candidates)
        diversity_boost, options = _candidate_diversity_signal(items, "material")
        self.assertEqual(len(items), 100)
        self.assertGreater(diversity_boost, 0.0)
        self.assertIn("polyester", options)
        self.assertNotIn("silk", options)

    def test_candidate_item_and_mapping_compatibility(self) -> None:
        candidate = _candidate(parent_asin="OBJECT", details={"Material": "Mesh"})
        self.assertEqual(_retrieval_items([candidate]), [candidate])
        self.assertEqual(_product(candidate)["parent_asin"], "OBJECT")

        mapping_candidate = {
            "item": {"parent_asin": "MAPPING", "details": {"Material": "Cotton"}},
            "retrieval_rank": 1,
        }
        self.assertEqual(_product(mapping_candidate)["parent_asin"], "MAPPING")

    def test_hard_and_soft_constraints_are_known(self) -> None:
        state = {
            "turn": 2,
            "hard_constraint": {"category": "running shoes", "budget": {"max": 100}},
            "soft_constraint": {"color": "black", "feature": ["lightweight"]},
        }
        decision = decide_ask(state, CANDIDATES_100)
        self.assertNotIn(decision["ask_attribute"], {"category", "budget", "color", "feature"})

    def test_attribute_value_object_marks_constraint_as_known(self) -> None:
        state = {
            "turn": 2,
            "hard_constraint": {
                "category": AttributeValueStub(),
                "color": AttributeValueStub(),
            },
        }
        decision = decide_ask(state, CANDIDATES_100)
        self.assertNotIn(decision["ask_attribute"], {"category", "color"})

    def test_no_prefernce_and_compatible_spelling_are_unavailable(self) -> None:
        self.assertEqual(
            _unavailable_attributes({"no_prefernce": ["brand", "color"]}),
            {"brand", "color"},
        )
        self.assertEqual(
            _unavailable_attributes({"no_preference": ["material"]}),
            {"material"},
        )
        self.assertEqual(
            _unavailable_attributes({"rejected_values": {"material": ["leather"]}}),
            set(),
        )

        known_except_brand_and_color = {
            attribute: "known"
            for attribute in (
                "category", "use_case", "feature", "size", "material",
                "budget", "style", "other",
            )
        }
        decision = decide_ask(
            {
                "turn": 2,
                "hard_constraint": known_except_brand_and_color,
                "no_prefernce": ["brand", "color"],
            },
            CANDIDATES_100,
        )
        self.assertIsNone(decision["ask_attribute"])

    def test_object_state_is_supported_and_can_record_question(self) -> None:
        state = SimpleNamespace(
            turn=2,
            hard_constraint={"category": "shoes"},
            soft_constraint={"feature": "comfortable"},
            no_prefernce=["brand"],
            asked_attributes=["size"],
            user_profile={"preference_tags": []},
        )
        decision = decide_ask(state, CANDIDATES_100)
        self.assertNotIn(decision["ask_attribute"], {"category", "feature", "brand", "size"})
        record_asked_attribute(state, decision["ask_attribute"])
        self.assertIn(decision["ask_attribute"], state.asked_attributes)

    def test_details_supply_material_and_color(self) -> None:
        product = {"details": {"Material": "Stainless Steel", "Color": "Silver"}}
        self.assertEqual(_values(product, "material"), {"stainless steel"})
        self.assertEqual(_values(product, "color"), {"silver"})

    def test_missing_details_uses_text_fallback(self) -> None:
        product = {"title": "Black cotton running shirt"}
        self.assertEqual(_values(product, "material"), {"cotton"})
        self.assertEqual(_values(product, "color"), {"black"})

    def test_brand_prefers_details_then_falls_back_to_store(self) -> None:
        self.assertEqual(
            _values({"details": {"Brand": "Acme"}, "store": "Other Store"}, "brand"),
            {"acme"},
        )
        self.assertEqual(_values({"store": "Other Store"}, "brand"), {"other store"})

    def test_public_entry_uses_module_1_candidates_only(self) -> None:
        self.assertEqual(
            list(signature(decide_ask).parameters),
            ["shopping_state", "candidates_100"],
        )

    def test_empty_candidates_and_missing_scores_are_safe(self) -> None:
        self.assertEqual(
            decide_ask({"turn": 1}, [])["ask_attribute"],
            "category",
        )
        decision = decide_ask(
            {"turn": 2, "hard_constraint": {"category": "shoes"}},
            [{"parent_asin": "A", "details": {"Color": "Black"}}],
        )
        self.assertIsNotNone(decision["ask_attribute"])


if __name__ == "__main__":
    unittest.main()
