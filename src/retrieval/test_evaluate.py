from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item import Candidate
from src.retrieval import Catalog
from src.retrieval.evaluate import evaluate_recall
from src.retrieval.experiment_hybrid import compare_hybrid_methods
from src.retrieval.experiment_text import run_text_ablation


PRODUCTS = [
    {
        "parent_asin": "TARGET-1",
        "title": "Black Cotton Shirt",
        "features": ["soft cotton"],
        "categories": ["Clothing", "Shirts"],
    },
    {
        "parent_asin": "TARGET-2",
        "title": "Trail Hiking Boots",
        "features": ["waterproof"],
        "categories": ["Shoes", "Hiking Boots"],
    },
    {
        "parent_asin": "OTHER",
        "title": "Other Product",
        "categories": ["Accessories"],
    },
]


class StaticRetriever:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> list[Candidate]:
        del query, state, intent
        return [
            Candidate(item=self.catalog["TARGET-1"], retrieval_score=1.0, retrieval_rank=1),
            Candidate(item=self.catalog["OTHER"], retrieval_score=0.5, retrieval_rank=2),
        ][:k]


class StaticDenseRetriever:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> list[Candidate]:
        del query, state, intent
        return [
            Candidate(item=self.catalog["TARGET-2"], dense_score=0.9, retrieval_score=0.9, retrieval_rank=1),
            Candidate(item=self.catalog["TARGET-1"], dense_score=0.7, retrieval_score=0.7, retrieval_rank=2),
        ][:k]


def _sample(sample_id: str, scenario: str, target: str) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "ground_truth": {"parent_asin": target},
        "user_profile": {},
    }


class RecallEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = Catalog.from_items(PRODUCTS)
        self.retriever = StaticRetriever(self.catalog)

    def test_recall_counts_exact_target_rank_and_scenarios(self) -> None:
        samples = [
            _sample("one", "buying", "TARGET-1"),
            _sample("two", "browsing", "TARGET-2"),
        ]
        result = evaluate_recall(
            self.retriever,
            self.catalog,
            samples,
            ks=(1, 2),
            include_sessions=True,
        )

        self.assertEqual(result["overall"]["sample_count"], 2)
        self.assertEqual(result["overall"]["hits_at_1"], 1)
        self.assertEqual(result["overall"]["recall_at_1"], 0.5)
        self.assertEqual(result["overall"]["recall_at_2"], 0.5)
        self.assertEqual(result["scenario_metrics"]["buying"]["recall_at_1"], 1.0)
        self.assertEqual(result["scenario_metrics"]["browsing"]["recall_at_2"], 0.0)
        self.assertEqual(result["sessions"][0]["target_rank"], 1)
        self.assertIsNone(result["sessions"][1]["target_rank"])
        self.assertIn("Shirts", result["sessions"][0]["query"])

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_recall(self.retriever, self.catalog, [], ks=(0,))

    def test_missing_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from the catalog"):
            evaluate_recall(
                self.retriever,
                self.catalog,
                [_sample("bad", "buying", "NOT-IN-CATALOG")],
            )

    def test_text_ablation_reports_each_requested_version(self) -> None:
        results = run_text_ablation(
            self.catalog,
            [_sample("one", "buying", "TARGET-1")],
            versions=("title_v0", "all_fields_v4"),
            ks=(1,),
        )
        self.assertEqual(
            [result["text_version"] for result in results],
            ["title_v0", "all_fields_v4"],
        )
        self.assertEqual(results[0]["fields"], ["title"])
        self.assertIn("recall_at_1", results[0]["overall"])

    def test_hybrid_comparison_reports_five_methods_and_union_semantics(self) -> None:
        samples = [
            _sample("one", "buying", "TARGET-1"),
            _sample("two", "browsing", "TARGET-2"),
        ]
        result = compare_hybrid_methods(
            self.retriever,
            StaticDenseRetriever(self.catalog),
            self.catalog,
            samples,
            ks=(1, 2),
            preload_dense_queries=False,
        )
        self.assertEqual(
            list(result["methods"]),
            ["bm25", "dense", "union", "rrf", "weighted"],
        )
        self.assertEqual(result["methods"]["union"]["overall"]["recall_at_1"], 1.0)
        self.assertIn("pool size <= 2K", result["config"]["union_semantics"])


if __name__ == "__main__":
    unittest.main()
