from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attribute import AttributeName, normalize_attribute_map
from src.item import Candidate
from src.retrieval import BM25Retriever, Catalog, Retriever, build_retrieval_query


PRODUCTS = [
    {
        "parent_asin": "RUN-1",
        "title": "Lightweight Road Running Shoes",
        "features": ["breathable mesh"],
        "categories": ["Shoes", "Running"],
        "price": 79.0,
    },
    {
        "parent_asin": "HIKE-1",
        "title": "Waterproof Hiking Boots",
        "features": ["ankle support", "trail grip"],
        "categories": ["Shoes", "Hiking Boots"],
        "price": 99.0,
    },
    {
        "parent_asin": "HIKE-2",
        "title": "Casual Hiking Shoes",
        "features": ["light trail use"],
        "categories": ["Shoes", "Hiking"],
        "price": None,
    },
]


class CatalogTest(unittest.TestCase):
    def test_load_normalizes_skips_malformed_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [
                json.dumps(PRODUCTS[0]),
                json.dumps({**PRODUCTS[0], "title": "duplicate"}),
                "not-json",
                json.dumps({"title": "missing identifier"}),
                json.dumps({"parent_asin": "MINIMAL", "features": None}),
            ]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            catalog = Catalog.load(path)

        self.assertEqual(list(catalog), ["RUN-1", "MINIMAL"])
        self.assertEqual(catalog["RUN-1"].title, PRODUCTS[0]["title"])
        self.assertEqual(catalog["MINIMAL"].features, [])
        self.assertEqual(catalog.stats.rows_seen, 5)
        self.assertEqual(catalog.stats.items_loaded, 2)
        self.assertEqual(catalog.stats.duplicate_asins, 1)
        self.assertEqual(catalog.stats.malformed_rows, 2)

    def test_gzip_catalog_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(PRODUCTS[0]) + "\n")
            catalog = Catalog.load(path)
        self.assertEqual(catalog["RUN-1"].parent_asin, "RUN-1")


class QueryConstructionTest(unittest.TestCase):
    def test_state_values_are_authoritative_for_intent_override(self) -> None:
        state = {
            "hard_constraint": {"category": "hiking boots"},
            "soft_constraint": {"feature": ["waterproof", "ankle support"]},
        }
        query = build_retrieval_query(
            "Actually forget running shoes; I need hiking boots.",
            state,
            "buying",
        )
        self.assertEqual(query, "hiking boots waterproof ankle support")
        self.assertNotIn("running", query)

    def test_no_preference_and_numeric_ranges_do_not_pollute_lexical_query(self) -> None:
        state = {
            "hard_constraint": normalize_attribute_map({
                "category": "running shoes",
                "budget": {"max": 100, "unit": "USD"},
            }),
            "soft_constraint": normalize_attribute_map(
                {"brand": "Example", "color": "black"}
            ),
            "no_prefernce": [AttributeName.BRAND],
        }
        self.assertEqual(build_retrieval_query("fallback", state), "running shoes black")

    def test_empty_state_falls_back_to_current_message(self) -> None:
        self.assertEqual(
            build_retrieval_query("  lightweight   shoes ", {"hard_constraint": {}}),
            "lightweight shoes",
        )


class BM25RetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = Catalog.from_items([*PRODUCTS, PRODUCTS[1]])
        self.retriever = BM25Retriever(self.catalog)

    def tearDown(self) -> None:
        self.retriever.close()

    def test_top_k_contract_order_scores_and_deduplication(self) -> None:
        candidates = self.retriever.retrieve("hiking", k=2)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(isinstance(value, Candidate) for value in candidates))
        self.assertEqual(len({value.parent_asin for value in candidates}), 2)
        self.assertEqual([value.retrieval_rank for value in candidates], [1, 2])
        self.assertGreaterEqual(candidates[0].retrieval_score, candidates[1].retrieval_score)
        self.assertEqual(candidates[0].bm25_score, candidates[0].retrieval_score)

    def test_state_drives_retrieval_after_override(self) -> None:
        state = {"hard_constraint": {"category": "hiking boots"}}
        candidates = self.retriever.retrieve("running shoes", state, "buying", k=1)
        self.assertEqual(candidates[0].parent_asin, "HIKE-1")

    def test_empty_query_and_non_positive_k_return_empty(self) -> None:
        self.assertEqual(self.retriever.retrieve("", k=10), [])
        self.assertEqual(self.retriever.retrieve("running", k=0), [])

    def test_output_is_deterministic(self) -> None:
        first = [value.parent_asin for value in self.retriever.retrieve("hiking", k=10)]
        second = [value.parent_asin for value in self.retriever.retrieve("hiking", k=10)]
        self.assertEqual(first, second)

    def test_facade_preserves_configurable_k(self) -> None:
        facade = Retriever(self.retriever)
        self.assertEqual(len(facade.retrieve("hiking", k=1)), 1)


if __name__ == "__main__":
    unittest.main()
