from __future__ import annotations

import unittest

from starter.intent import classify_intent


class IntentClassifierTest(unittest.TestCase):
    def test_detects_explicit_buying_intent(self) -> None:
        result = classify_intent("Find me running shoes under $120")

        self.assertEqual(result.intent, "buying")
        self.assertGreaterEqual(result.confidence, 0.8)

    def test_detects_explicit_browsing_intent(self) -> None:
        result = classify_intent("I'm just browsing for summer style ideas")

        self.assertEqual(result.intent, "browsing")

    def test_constraint_follow_up_inherits_previous_intent(self) -> None:
        result = classify_intent("Preferably black", previous_intent="buying")

        self.assertEqual(result.intent, "buying")
        self.assertIn("inherited from previous intent", result.evidence)

    def test_detects_intent_override(self) -> None:
        result = classify_intent(
            "Actually, forget running shoes. Now I need hiking boots.",
            previous_intent="buying",
        )

        self.assertEqual(result.intent, "buying")
        self.assertTrue(result.is_override)

    def test_returns_unknown_without_clear_signal_or_context(self) -> None:
        result = classify_intent("Hello there")

        self.assertEqual(result.intent, "unknown")
        self.assertLess(result.confidence, 0.5)

    def test_empty_message_is_unknown(self) -> None:
        result = classify_intent("   ", previous_intent="buying")

        self.assertEqual(result.intent, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_mixed_buying_and_browsing_signals_are_marked_as_conflict(self) -> None:
        result = classify_intent("I want ideas for black shoes")

        self.assertTrue(result.is_conflict)
        self.assertFalse(result.is_clear)
        self.assertIn("stated need", result.evidence)
        self.assertIn("ideas or inspiration", result.evidence)


if __name__ == "__main__":
    unittest.main()
