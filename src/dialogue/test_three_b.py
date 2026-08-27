from __future__ import annotations

import unittest

from src.dialogue import build_question, decide_ask, record_asked_attribute


RANKING_RESULT = {
    "results": [
        {"parent_asin": "A", "product": {"category": "shoes", "material": "leather", "color": "black", "size": "wide"}},
        {"parent_asin": "B", "product": {"category": "shoes", "material": "canvas", "color": "white", "size": "regular"}},
        {"parent_asin": "C", "product": {"category": "shoes", "material": "leather", "color": "brown", "size": "regular"}},
    ]
}


class AskAttributeSelectorTest(unittest.TestCase):
    def test_unknown_category_is_asked_first(self) -> None:
        decision = decide_ask({"turn": 1}, RANKING_RESULT)
        self.assertEqual(decision["ask_attribute"], "category")

    def test_never_repeats_known_asked_or_unavailable_attributes(self) -> None:
        state = {
            "turn": 3,
            "known_attributes": {"category": "shoes", "use_case": "walking"},
            "asked_attributes": ["feature", "size"],
            "no_preference_attributes": ["material"],
        }
        decision = decide_ask(state, RANKING_RESULT)
        self.assertNotIn(
            decision["ask_attribute"],
            {"category", "use_case", "feature", "size", "material"},
        )

    def test_known_attributes_supports_list_format(self) -> None:
        state = {
            "turn": 1,
            "known_attributes": ["category", "color"],
        }
        decision = decide_ask(state, RANKING_RESULT)
        self.assertNotIn(decision["ask_attribute"], {"category", "color"})

    def test_ranking_ambiguity_affects_selection(self) -> None:
        state = {
            "turn": 3,
            "known_attributes": {"category": "shoes", "use_case": "walking", "feature": "comfortable"},
            "asked_attributes": ["size"],
            "user_profile": {"preference_tags": ["material"]},
        }
        decision = decide_ask(state, RANKING_RESULT)
        self.assertEqual(decision["ask_attribute"], "material")
        self.assertIn("leather", decision["message"])

    def test_turn_ten_does_not_ask_another_question(self) -> None:
        decision = decide_ask({"turn": 10}, RANKING_RESULT)
        self.assertEqual(decision, {"ask_attribute": None, "message": ""})

    def test_size_question_uses_ranking_options(self) -> None:
        question = build_question("size", ["small", "medium", "large"])
        self.assertIn("small, medium, or large", question)

    def test_caller_can_record_current_question_in_state(self) -> None:
        state = {"asked_attributes": ["category"]}
        record_asked_attribute(state, "size")
        self.assertEqual(state["asked_attributes"], ["category", "size"])

    def test_invalid_turn_falls_back_without_exception(self) -> None:
        decision = decide_ask({"turn": "second"}, RANKING_RESULT)
        self.assertEqual(decision["ask_attribute"], "category")


if __name__ == "__main__":
    unittest.main()
