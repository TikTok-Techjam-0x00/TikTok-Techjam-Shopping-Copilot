from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval import (
    Catalog,
    HybridRetriever,
    IntentRoutedRetriever,
    Retriever,
    build_embedding_cache,
)


PRODUCTS = [
    {
        "parent_asin": "RUN-1",
        "title": "Lightweight Road Running Shoes",
        "categories": ["Shoes", "Running"],
    },
    {
        "parent_asin": "HIKE-1",
        "title": "Waterproof Hiking Boots",
        "categories": ["Shoes", "Hiking Boots"],
    },
]


class FakeEmbeddingEncoder:
    model = "fake-semantic-v1"
    dimension = 2
    batch_size = 2

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [
                    float("running" in text.casefold()),
                    float("hiking" in text.casefold()),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_queries(
        self,
        texts: list[str],
        *,
        instruct: str | None = None,
    ) -> np.ndarray:
        del instruct
        return self.encode(texts)


class SOTADefaultHybridTest(unittest.TestCase):
    def test_factory_builds_measured_outer_routed_hybrid(self) -> None:
        encoder = FakeEmbeddingEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in PRODUCTS),
                encoding="utf-8",
            )
            cache_dir = root / "cache"
            cache = build_embedding_cache(
                Catalog.load(catalog_path),
                encoder,
                cache_dir,
                text_version="all_fields_v4",
            )
            cache.close()

            retriever = Retriever.intent_routed_hybrid_weighted(
                str(catalog_path),
                encoder,
                cache_dir,
            )
            strategy = retriever.strategy
            self.assertIsInstance(strategy, IntentRoutedRetriever)
            self.assertIsInstance(strategy.buying, HybridRetriever)
            self.assertEqual(strategy.buying.config.method, "weighted")
            self.assertEqual(strategy.buying.config.alpha, 0.7)
            self.assertEqual(strategy.buying.config.source_k, 200)
            self.assertEqual(
                strategy.route_name({"intent": "browsing", "turn": 1}),
                "browsing",
            )
            self.assertEqual(
                strategy.route_name({"intent": "browsing", "turn": 2}),
                "buying",
            )

            first_turn = retriever.retrieve(
                "running shoes",
                state={"intent": "browsing", "turn": 1},
                k=1,
            )
            buying = retriever.retrieve(
                "hiking boots",
                state={"intent": "buying", "turn": 1},
                k=1,
            )
            self.assertEqual(first_turn[0].parent_asin, "RUN-1")
            self.assertEqual(buying[0].parent_asin, "HIKE-1")

            strategy.buying.bm25.close()
            strategy.buying.dense.close()
            strategy.browsing.close()


if __name__ == "__main__":
    unittest.main()
