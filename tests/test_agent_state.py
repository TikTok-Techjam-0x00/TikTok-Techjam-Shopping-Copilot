from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class StatefulAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        catalog = Path(self.temporary_directory.name) / "catalog.jsonl"
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
        self.assertEqual(state["category"], "black hiking boots")
        self.assertEqual(state["hard_constraints"]["color"], "black")
        self.assertEqual(state["turn_count"], 1)
        self.assertIn(response["ask_attribute"], state["asked_attributes"])

    def test_sessions_are_isolated(self) -> None:
        self.agent.reset("first", {})
        self.agent.reset("second", {})

        self.agent.respond("first", "I want running shoes.", 1, 10)

        self.assertEqual(self.agent.get_state("first")["category"], "running shoes")
        self.assertIsNone(self.agent.get_state("second")["category"])

    def test_respond_requires_reset(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "I want shoes", 1, 10)


if __name__ == "__main__":
    unittest.main()
