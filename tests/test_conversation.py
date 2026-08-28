from __future__ import annotations

import json
import unittest

from src.attribute import AttributeName
from src.dialogue import decide_ask
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

    def test_browsing_to_buying_is_an_implicit_intent_override(self) -> None:
        state = create_state("session")
        update_state(state, "I'm still exploring running shoes.", turn=1)
        update_state(state, "I need black hiking boots under $100.", turn=2)

        self.assertEqual(state.intent, "buying")
        self.assertTrue(state.override_detected)
        self.assertEqual(state.category, "black hiking boots")
        self.assertEqual(state.hard_constraint[AttributeName.BUDGET].maximum, 100.0)
        self.assertNotIn("running", retrieval_query(state))
        self.assertEqual(
            state.intent_transitions[-1],
            {"turn": 2, "from": "browsing", "to": "buying"},
        )

    def test_buying_to_browsing_clears_purchase_constraints(self) -> None:
        state = create_state("session")
        update_state(state, "I need black running shoes under $120.", turn=1)
        update_state(
            state,
            "Actually, forget the requirements. I'm just browsing. Show me winter jackets.",
            turn=2,
        )

        self.assertEqual(state.intent, "browsing")
        self.assertTrue(state.override_detected)
        self.assertEqual(state.category, "winter jackets")
        self.assertNotIn(AttributeName.BUDGET, state.hard_constraint)
        self.assertNotIn(AttributeName.COLOR, state.hard_constraint)
        self.assertNotIn("running", retrieval_query(state))

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
        self.assertTrue(state.boundary_detected)
        self.assertIn(AttributeName.COLOR, state.boundary_attributes)

    def test_boundary_reply_uses_the_attribute_from_the_previous_question(self) -> None:
        state = create_state("session")
        update_state(state, "I need hiking boots.", turn=1)
        state.mark_attribute_asked("material")

        update_state(
            state,
            "It doesn't matter to me; use your judgment.",
            turn=2,
            asked_attribute="material",
        )
        decision = decide_ask(state, [])

        self.assertIn(AttributeName.MATERIAL, state.no_prefernce)
        self.assertNotEqual(decision["ask_attribute"], "material")

    def test_explicit_any_value_boundary_is_recorded(self) -> None:
        state = create_state("session")
        update_state(state, "Any color is fine.", turn=2, asked_attribute="color")

        self.assertIn(AttributeName.COLOR, state.no_prefernce)
        self.assertTrue(state.boundary_detected)

    def test_explicit_do_not_consider_boundary_is_recorded(self) -> None:
        state = create_state("session")
        update_state(
            state,
            "I do not want to consider size at all.",
            turn=2,
            asked_attribute="size",
        )

        self.assertIn(AttributeName.SIZE, state.no_prefernce)
        self.assertIn(AttributeName.SIZE, state.boundary_attributes)

    def test_product_text_with_any_other_is_not_a_boundary(self) -> None:
        state = create_state("session")
        update_state(
            state,
            "For that, what matters is: suitable for home and any other casual occasion.",
            turn=2,
            asked_attribute="style",
        )

        self.assertFalse(state.boundary_detected)
        self.assertEqual(state.no_prefernce, [])
        self.assertIn(AttributeName.FEATURE, state.soft_constraint)

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
