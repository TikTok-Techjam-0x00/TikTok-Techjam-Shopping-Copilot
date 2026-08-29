from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.attribute import AttributeName
from src.state import QwenSemanticResolver, SemanticRequest


class FakeCompletions:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class QwenSemanticResolverTest(unittest.TestCase):
    def test_returns_normalized_state_update_from_json(self) -> None:
        completions = FakeCompletions({
            "intent": "buying",
            "soft_constraint": {"style": "more formal"},
            "override": True,
            "confidence": 0.87,
        })
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        resolver = QwenSemanticResolver(api_key="test-key", client=client)
        request = SemanticRequest(
            message="Make it more formal.",
            current_state={},
            recent_history=("I need a jacket.",),
            asked_attribute=None,
            rule_intent="buying",
            rule_intent_confidence=0.5,
            rule_evidence=(),
            fallback_reasons=("unresolved_semantic_comparison",),
        )

        result = resolver.resolve(request)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.update.intent, "buying")
        self.assertEqual(
            result.update.soft_constraint[AttributeName.STYLE].values,
            ["more formal"],
        )
        self.assertTrue(result.update.override)
        self.assertEqual(result.confidence, 0.87)
        self.assertEqual(completions.calls[0]["temperature"], 0)
        self.assertEqual(completions.calls[0]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
