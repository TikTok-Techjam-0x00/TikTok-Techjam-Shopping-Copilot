from __future__ import annotations

import unittest

from src.attribute import AttributeName
from src.state import (
    CallableSemanticResolver,
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


if __name__ == "__main__":
    unittest.main()
