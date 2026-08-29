from __future__ import annotations

import unittest

from src.attribute import AttributeName
from src.state import (
    CallableSemanticResolver,
    SemanticPolicy,
    SemanticRequest,
    SemanticResolution,
    StateUpdate,
    create_state,
    update_state,
)


class RecordingResolver:
    def __init__(self, resolution: SemanticResolution | None) -> None:
        self.resolution = resolution
        self.requests: list[SemanticRequest] = []

    def resolve(self, request: SemanticRequest) -> SemanticResolution | None:
        self.requests.append(request)
        return self.resolution


class FailingResolver:
    def resolve(self, request: SemanticRequest) -> SemanticResolution | None:
        raise RuntimeError("semantic service unavailable")


class HybridStateManagerTest(unittest.TestCase):
    def test_simple_deterministic_message_never_calls_semantic_resolver(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(StateUpdate.from_raw(hard_constraint={"style": "formal"}), 0.9)
        )
        state = create_state("session")

        update_state(
            state,
            "Find me black running shoes under $100.",
            turn=1,
            semantic_resolver=resolver,
        )

        self.assertEqual(resolver.requests, [])
        self.assertFalse(state.semantic_fallback_used)
        self.assertEqual(state.hard_constraint[AttributeName.COLOR].values, ["black"])
        self.assertEqual(state.hard_constraint[AttributeName.BUDGET].maximum, 100.0)

    def test_context_dependent_message_uses_semantic_resolver(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(
                StateUpdate.from_raw(
                    intent="buying",
                    hard_constraint={"category": "blazer"},
                    soft_constraint={"style": "formal"},
                ),
                0.88,
            )
        )
        state = create_state("session")
        update_state(state, "I need a jacket.", turn=1)

        update_state(
            state,
            "Something like the last one, but more formal.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(len(resolver.requests), 1)
        self.assertIn("context_dependent_reference", resolver.requests[0].fallback_reasons)
        self.assertEqual(state.category, "blazer")
        self.assertEqual(state.soft_constraint[AttributeName.STYLE].values, ["formal"])
        self.assertTrue(state.semantic_fallback_used)
        self.assertEqual(state.semantic_fallback_count, 1)

    def test_rule_values_win_when_semantic_fallback_adds_context(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(
                StateUpdate.from_raw(
                    hard_constraint={"color": "white"},
                    soft_constraint={"style": "formal"},
                ),
                0.9,
            )
        )
        state = create_state("session")
        update_state(state, "I need casual shoes.", turn=1)

        update_state(
            state,
            "I need black shoes like the last one.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(len(resolver.requests), 1)
        self.assertEqual(state.hard_constraint[AttributeName.COLOR].values, ["black"])
        self.assertEqual(state.soft_constraint[AttributeName.STYLE].values, ["formal"])

    def test_missing_resolver_keeps_rule_only_fallback(self) -> None:
        state = create_state("session")

        update_state(state, "Something similar to the previous one.", turn=1)

        self.assertFalse(state.semantic_fallback_used)
        self.assertIn("context_dependent_reference", state.semantic_fallback_reasons)

    def test_resolver_failure_never_breaks_state_update(self) -> None:
        state = create_state("session")

        update_state(
            state,
            "Something similar to the previous one.",
            turn=1,
            semantic_resolver=FailingResolver(),
        )

        self.assertTrue(state.semantic_fallback_used)
        self.assertEqual(state.semantic_fallback_count, 1)
        self.assertEqual(state.turn, 1)

    def test_callable_adapter_normalizes_structured_llm_output(self) -> None:
        resolver = CallableSemanticResolver(lambda request: {
            "intent": "buying",
            "hard_constraint": {"category": "business casual blazer"},
            "soft_constraint": {"style": ["minimal", "formal"]},
            "confidence": 0.91,
        })
        state = create_state("session")

        update_state(
            state,
            "Something more polished than the previous one.",
            turn=1,
            semantic_resolver=resolver,
        )

        self.assertEqual(state.category, "business casual blazer")
        self.assertEqual(state.soft_constraint[AttributeName.STYLE].values, ["minimal", "formal"])
        self.assertEqual(state.intent_confidence, 0.91)

    def test_rule_conflict_routes_to_semantic_resolver_even_with_slots(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(StateUpdate.from_raw(intent="browsing"), 0.90)
        )
        state = create_state("session")

        update_state(
            state,
            "I want ideas for black shoes.",
            turn=1,
            semantic_resolver=resolver,
        )

        self.assertEqual(len(resolver.requests), 1)
        self.assertIn("conflicting_rule_signals", resolver.requests[0].fallback_reasons)
        self.assertEqual(state.intent, "browsing")
        self.assertEqual(state.intent_resolution_source, "llm")

    def test_history_smoothing_rejects_one_weak_ambiguous_intent_flip(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(StateUpdate.from_raw(intent="browsing"), 0.75)
        )
        state = create_state("session")
        update_state(state, "I need running shoes.", turn=1)

        update_state(
            state,
            "I want ideas for a different style.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.intent_resolution_source, "history")
        self.assertTrue(state.intent_smoothed)
        self.assertEqual(state.intent_transitions, [])

    def test_high_confidence_semantic_intent_can_change_history(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(StateUpdate.from_raw(intent="browsing"), 0.91)
        )
        state = create_state("session")
        update_state(state, "I need running shoes.", turn=1)

        update_state(
            state,
            "I want ideas for a different style.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(state.intent, "browsing")
        self.assertEqual(state.intent_resolution_source, "llm")
        self.assertFalse(state.intent_smoothed)
        self.assertEqual(
            state.intent_transitions[-1],
            {"turn": 2, "from": "buying", "to": "browsing"},
        )

    def test_low_confidence_semantic_result_fails_validation_before_merge(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(
                StateUpdate.from_raw(intent="browsing", hard_constraint={"color": "white"}),
                0.40,
            )
        )
        state = create_state("session")
        update_state(state, "I need black running shoes.", turn=1)

        update_state(
            state,
            "Maybe something else.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.hard_constraint[AttributeName.COLOR].values, ["black"])
        self.assertIn(
            "semantic_confidence_below_threshold",
            state.semantic_validation_errors,
        )

    def test_semantic_control_requires_explicit_change_confidence(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(
                StateUpdate.from_raw(
                    intent="browsing",
                    override=True,
                    clear_hard_constraint=True,
                ),
                0.60,
            )
        )
        state = create_state("session")
        update_state(state, "I need black running shoes under $100.", turn=1)

        update_state(
            state,
            "Maybe something else.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(state.intent, "buying")
        self.assertIn(AttributeName.CATEGORY, state.hard_constraint)
        self.assertIn(AttributeName.BUDGET, state.hard_constraint)
        self.assertIn(
            "semantic_control_below_threshold",
            state.semantic_validation_errors,
        )

    def test_clear_rule_blocks_conflicting_semantic_control_flags(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(
                StateUpdate.from_raw(
                    intent="browsing",
                    override=True,
                    clear_hard_constraint=True,
                    clear_soft_constraint=True,
                ),
                0.95,
            )
        )
        state = create_state("session")
        update_state(state, "I need running shoes under $100.", turn=1)

        update_state(
            state,
            "Find me a jacket similar to the previous one.",
            turn=2,
            semantic_resolver=resolver,
        )

        self.assertEqual(len(resolver.requests), 1)
        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.intent_resolution_source, "rule")
        self.assertIn(AttributeName.CATEGORY, state.hard_constraint)
        self.assertIn(AttributeName.BUDGET, state.hard_constraint)
        self.assertFalse(state.override_detected)

    def test_semantic_no_preference_is_recorded_as_a_boundary(self) -> None:
        resolver = CallableSemanticResolver(lambda request: {
            "intent": "browsing",
            "no_preference": ["style"],
            "confidence": 0.9,
        })
        state = create_state("session")

        update_state(
            state,
            "Whichever look makes sense.",
            turn=1,
            semantic_resolver=resolver,
        )

        self.assertTrue(state.boundary_detected)
        self.assertIn(AttributeName.STYLE, state.no_prefernce)
        self.assertIn(AttributeName.STYLE, state.boundary_attributes)

    def test_callable_adapter_rejects_non_schema_fields(self) -> None:
        resolver = CallableSemanticResolver(lambda request: {
            "intent": "buying",
            "confidence": 0.9,
            "reasoning": "unsupported free-form field",
        })
        state = create_state("session")

        update_state(
            state,
            "Something similar to the previous one.",
            turn=1,
            semantic_resolver=resolver,
        )

        self.assertEqual(state.intent_resolution_source, "default")
        self.assertIn("empty_or_invalid_semantic_result", state.semantic_validation_errors)

    def test_semantic_request_contains_only_bounded_recent_history(self) -> None:
        resolver = RecordingResolver(
            SemanticResolution(StateUpdate.from_raw(intent="buying"), 0.90)
        )
        state = create_state("session")
        state.intent = "buying"
        state.intent_confidence = 0.8
        state.history.extend([f"turn {number}" for number in range(1, 7)])

        update_state(
            state,
            "Something similar to the previous one.",
            turn=7,
            semantic_resolver=resolver,
            semantic_policy=SemanticPolicy(recent_history_turns=3),
        )

        self.assertEqual(resolver.requests[0].recent_history, ("turn 4", "turn 5", "turn 6"))
        self.assertNotIn("history", resolver.requests[0].current_state)
        self.assertNotIn("user_message", resolver.requests[0].current_state)


if __name__ == "__main__":
    unittest.main()
