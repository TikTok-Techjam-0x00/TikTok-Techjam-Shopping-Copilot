from __future__ import annotations

import unittest

from src.attribute import AttributeName
from src.state import ShoppingState, StateUpdate


class ShoppingStateTest(unittest.TestCase):
    def test_sessions_do_not_share_mutable_state(self) -> None:
        first = ShoppingState("first")
        second = ShoppingState("second")
        first.apply_update(StateUpdate.from_raw(hard_constraint={"color": "blue"}))
        self.assertEqual(second.hard_constraint, {})

    def test_accumulates_constraints_across_updates(self) -> None:
        state = ShoppingState("session")
        state.apply_update(StateUpdate.from_raw(hard_constraint={"category": "running shoes"}))
        state.apply_update(StateUpdate.from_raw(hard_constraint={"budget": {"max": 120, "unit": "USD"}}))
        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.hard_constraint[AttributeName.BUDGET].maximum, 120.0)

    def test_override_replaces_category_and_stale_use_case(self) -> None:
        state = ShoppingState("session")
        state.apply_update(StateUpdate.from_raw(hard_constraint={"category": "running shoes", "use_case": "running"}))
        state.apply_update(StateUpdate.from_raw(hard_constraint={"category": "hiking boots", "use_case": "hiking"}, override=True))
        self.assertEqual(state.category, "hiking boots")
        self.assertEqual(state.hard_constraint[AttributeName.USE_CASE].values, ["hiking"])

    def test_no_preference_removes_constraint(self) -> None:
        state = ShoppingState("session")
        state.apply_update(StateUpdate.from_raw(hard_constraint={"material": "cotton"}))
        state.apply_update(StateUpdate.from_raw(no_preference={"material"}))
        self.assertNotIn(AttributeName.MATERIAL, state.hard_constraint)
        self.assertIn(AttributeName.MATERIAL, state.no_prefernce)

    def test_new_value_restores_no_preference_attribute(self) -> None:
        state = ShoppingState("session", no_prefernce=[AttributeName.COLOR])
        state.apply_update(StateUpdate.from_raw(hard_constraint={"color": "black"}))
        self.assertEqual(state.hard_constraint[AttributeName.COLOR].values, ["black"])
        self.assertNotIn(AttributeName.COLOR, state.no_prefernce)

    def test_override_starts_a_new_epoch_and_preserves_provenance(self) -> None:
        state = ShoppingState("session")
        state.apply_update(
            StateUpdate.from_raw(
                hard_constraint={"category": "running shoes", "color": "blue"},
            ),
            source_turn=1,
        )
        state.mark_attribute_asked("material")

        state.apply_update(
            StateUpdate.from_raw(
                hard_constraint={"material": "leather"},
                override=True,
            ),
            source_turn=2,
        )

        material = state.constraint_provenance["hard_constraint"][
            AttributeName.MATERIAL
        ]
        self.assertEqual(state.constraint_epoch, 1)
        self.assertEqual(state.last_override_turn, 2)
        self.assertEqual(material.source_turn, 2)
        self.assertEqual(material.constraint_epoch, 1)
        self.assertEqual(material.confidence, 1.0)
        self.assertEqual(material.value.values, ["leather"])
        self.assertEqual(state.asked_attributes, [])
        self.assertEqual(state.asked_attributes_by_epoch[0], ["material"])

    def test_superseded_constraint_is_kept_in_history(self) -> None:
        state = ShoppingState("session")
        state.apply_update(
            StateUpdate.from_raw(hard_constraint={"material": "cotton"}),
            source_turn=1,
        )
        state.apply_update(
            StateUpdate.from_raw(hard_constraint={"material": "leather"}),
            source_turn=2,
        )

        self.assertEqual(state.hard_constraint[AttributeName.MATERIAL].values, ["leather"])
        self.assertEqual(state.constraint_history[-1].value.values, ["cotton"])


if __name__ == "__main__":
    unittest.main()
