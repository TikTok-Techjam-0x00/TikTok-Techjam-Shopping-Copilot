from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from src.state import SemanticResolution, StateUpdate


class StaticSemanticResolver:
    def __init__(self) -> None:
        self.call_count = 0

    def resolve(self, request: object) -> SemanticResolution:
        self.call_count += 1
        return SemanticResolution(
            StateUpdate.from_raw(
                intent="buying",
                hard_constraint={"category": "hiking boots"},
            ),
            0.9,
        )


class StatefulAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog = Path(self.temporary_directory.name) / "catalog.jsonl"
        self.catalog = catalog
        products = [
            {
                "parent_asin": "RUN1",
                "title": "Blue running shoes",
                "categories": ["Shoes", "Running"],
                "features": ["Lightweight mesh"],
                "details": {},
                "store": "Example",
                "description": [],
            },
            {
                "parent_asin": "HIKE1",
                "title": "Black hiking boots",
                "categories": ["Shoes", "Hiking Boots"],
                "features": ["Waterproof outdoor boot"],
                "details": {},
                "store": "Example",
                "description": [],
            },
        ]
        catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reset_respond_and_pipeline_state(self) -> None:
        self.agent.reset("session-a", {"preference_tags": ["fit"]})

        response = self.agent.respond(
            "session-a",
            "Find me black hiking boots.",
            turn=1,
            top_k=10,
        )
        state = self.agent.get_state("session-a")

        self.assertEqual(response["recommendations"][0]["parent_asin"], "HIKE1")
        self.assertEqual(state["hard_constraint"]["category"]["values"], ["black hiking boots"])
        self.assertEqual(state["hard_constraint"]["color"]["values"], ["black"])
        self.assertEqual(state["turn"], 1)
        self.assertIn(response["ask_attribute"], state["asked_attributes"])

    def test_sessions_are_isolated(self) -> None:
        self.agent.reset("first", {})
        self.agent.reset("second", {})

        self.agent.respond("first", "I want running shoes.", 1, 10)

        self.assertEqual(
            self.agent.get_state("first")["hard_constraint"]["category"]["values"],
            ["running shoes"],
        )
        self.assertNotIn("category", self.agent.get_state("second")["hard_constraint"])

    def test_respond_requires_reset(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "I want shoes", 1, 10)

    def test_agent_passes_ambiguous_context_to_semantic_resolver(self) -> None:
        resolver = StaticSemanticResolver()
        agent = Agent(self.catalog, semantic_resolver=resolver)
        agent.reset("semantic", {})

        response = agent.respond(
            "semantic",
            "Something like the previous one, but suitable for hiking.",
            1,
            10,
        )
        state = agent.get_state("semantic")

        self.assertEqual(resolver.call_count, 1)
        self.assertTrue(state["semantic_fallback_used"])
        self.assertEqual(
            response["recommendations"][0]["parent_asin"],
            "HIKE1",
        )


if __name__ == "__main__":
    unittest.main()
