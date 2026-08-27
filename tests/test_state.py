from __future__ import annotations

import unittest

from starter.state import ShoppingState, StateUpdate


class ShoppingStateTest(unittest.TestCase):
    def test_sessions_do_not_share_mutable_state(self) -> None:
        first = ShoppingState()
        second = ShoppingState()

        first.add_hard_constraint("color", "blue")
        first.add_soft_preference("comfortable")

        self.assertEqual(second.hard_constraints, {})
        self.assertEqual(second.soft_preferences, [])

    def test_accumulates_constraints_across_updates(self) -> None:
        state = ShoppingState()

        state.apply_update(StateUpdate(category="running shoes"))
        state.apply_update(StateUpdate(hard_constraints={"price_max": 120.0}))

        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.hard_constraints["price_max"], 120.0)

    def test_new_category_replaces_old_category(self) -> None:
        state = ShoppingState(category="running shoes")

        state.apply_update(StateUpdate(category="hiking boots", override=True))

        self.assertEqual(state.category, "hiking boots")

    def test_rejected_attribute_removes_constraint(self) -> None:
        state = ShoppingState(hard_constraints={"material": "cotton"})

        state.apply_update(StateUpdate(rejected_attributes={"material"}))

        self.assertNotIn("material", state.hard_constraints)
        self.assertIn("material", state.rejected_attributes)

    def test_new_value_restores_rejected_attribute(self) -> None:
        state = ShoppingState(rejected_attributes={"color"})

        state.apply_update(StateUpdate(hard_constraints={"color": "black"}))

        self.assertEqual(state.hard_constraints["color"], "black")
        self.assertNotIn("color", state.rejected_attributes)


if __name__ == "__main__":
    unittest.main()
