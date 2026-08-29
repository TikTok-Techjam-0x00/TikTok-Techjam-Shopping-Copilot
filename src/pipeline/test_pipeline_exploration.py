from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.item import Candidate, Item
from src.pipeline import Pipeline


class _StaticRetriever:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def retrieve_page(self, *args: object, **kwargs: object) -> list[Candidate]:
        return list(self.candidates)

    def retrieve_strata(self, *args: object, **kwargs: object) -> list[Candidate]:
        return list(self.candidates)


class PipelineExplorationTest(unittest.TestCase):
    def setUp(self) -> None:
        products = [
            {"parent_asin": "A", "title": "Alpha"},
            {"parent_asin": "B", "title": "Beta"},
            {"parent_asin": "C", "title": "Gamma"},
        ]
        self.directory = tempfile.TemporaryDirectory()
        catalog = Path(self.directory.name) / "catalog.jsonl"
        catalog.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.pipeline = Pipeline(catalog, semantic_resolver=None)
        self._original_strategy = self.pipeline.retriever.strategy
        self.pipeline.retriever = _StaticRetriever([
            Candidate(item=Item(parent_asin="A", title="Alpha")),
            Candidate(item=Item(parent_asin="B", title="Beta")),
            Candidate(item=Item(parent_asin="C", title="Gamma")),
        ])
        self.pipeline.reset("session", {})

    def tearDown(self) -> None:
        pending = [self._original_strategy]
        closed: set[int] = set()
        while pending:
            strategy = pending.pop()
            if id(strategy) in closed:
                continue
            closed.add(id(strategy))
            for name in ("buying", "browsing", "bm25", "dense"):
                child = getattr(strategy, name, None)
                if child is not None:
                    pending.append(child)
            close = getattr(strategy, "close", None)
            if callable(close):
                close()
        self.directory.cleanup()

    def test_continued_session_does_not_repeat_an_exact_product(self) -> None:
        first = self.pipeline.respond(
            "session",
            "I'm looking for accessories, but I'm still exploring.",
            1,
            10,
        )
        second = self.pipeline.respond(
            "session",
            "I don't have an additional preference for other.",
            2,
            10,
        )

        self.assertEqual(first["recommendations"][0]["parent_asin"], "A")
        self.assertEqual(second["recommendations"][0]["parent_asin"], "B")

    def test_intent_override_resets_recommendation_memory(self) -> None:
        first = self.pipeline.respond(
            "session",
            "I'm looking for accessories, but I'm still exploring.",
            1,
            10,
        )
        overridden = self.pipeline.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: accessories.",
            2,
            10,
        )

        self.assertEqual(first["recommendations"][0]["parent_asin"], "A")
        self.assertEqual(overridden["recommendations"][0]["parent_asin"], "A")


if __name__ == "__main__":
    unittest.main()
