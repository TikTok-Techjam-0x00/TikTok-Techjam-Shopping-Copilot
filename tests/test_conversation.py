from __future__ import annotations

import json
import unittest

from src.attribute import AttributeName
from src.state import StateUpdate
from src.state import (
    create_state,
    retrieval_query,
    sanitize_retrieval_text,
    update_state,
)


class ConversationStateTest(unittest.TestCase):
    def test_accumulates_and_overrides_category_without_losing_budget(self) -> None:
        state = create_state("session")

        update_state(state, "I want running shoes.", turn=1)
        update_state(state, "Under $120.", turn=2)
        update_state(
            state,
            "Actually, forget running shoes. Now I need hiking boots.",
            turn=3,
        )

        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.category, "hiking boots")
        self.assertEqual(state.hard_constraint[AttributeName.BUDGET].maximum, 120.0)
        self.assertEqual(state.hard_constraint[AttributeName.USE_CASE].values, ["hiking"])
        self.assertNotIn("running", retrieval_query(state))
        self.assertTrue(state.override_detected)

    def test_browsing_prompt_routes_to_browsing(self) -> None:
        state = create_state("session")

        update_state(state, "I'm looking for women's shoes, but I'm still exploring.")

        self.assertEqual(state.intent, "browsing")
        self.assertEqual(state.category, "women's shoes")

    def test_direct_answer_uses_previous_asked_attribute(self) -> None:
        state = create_state("session")

        update_state(state, "Something for winter.", asked_attribute="use_case")

        self.assertEqual(state.hard_constraint[AttributeName.USE_CASE].values, ["winter"])

    def test_no_preference_rejects_and_removes_slot(self) -> None:
        state = create_state("session")
        state.apply_update(StateUpdate.from_raw(hard_constraint={"color": "blue"}))

        update_state(
            state,
            "I don't have a preference for color; please use your judgment.",
            asked_attribute="color",
        )

        self.assertNotIn(AttributeName.COLOR, state.hard_constraint)
        self.assertIn(AttributeName.COLOR, state.no_prefernce)

    def test_negative_constraint_is_not_a_positive_requirement(self) -> None:
        state = create_state("session")

        update_state(state, "I need a jacket, but not leather.")

        self.assertNotIn(AttributeName.MATERIAL, state.hard_constraint)
        self.assertEqual(state.rejected_values[AttributeName.MATERIAL].values, ["leather"])

    def test_pipeline_contract_is_json_serializable(self) -> None:
        state = create_state("session", {"preference_tags": ["fit"]})
        update_state(state, "Find me black running shoes under $100", turn=1)

        payload = state.to_dict()

        json.dumps(payload)
        self.assertEqual(payload["hard_constraint"]["category"]["values"], ["black running shoes"])
        self.assertEqual(payload["hard_constraint"]["budget"]["max"], 100.0)
        self.assertEqual(payload["asked_attributes"], [])

    def test_retrieval_text_never_contains_cjk_content(self) -> None:
        state = create_state("session")
        state.apply_update(StateUpdate.from_raw(
            hard_constraint={"category": "running shoes " + "\u767b\u5c71\u978b"},
            soft_constraint={"feature": "waterproof " + "\u9632\u6c34"},
        ))

        query = retrieval_query(state)

        self.assertEqual(query, "running shoes waterproof")
        self.assertEqual(sanitize_retrieval_text("\u5e2e\u6211\u627e hiking boots"), "hiking boots")


if __name__ == "__main__":
    unittest.main()
