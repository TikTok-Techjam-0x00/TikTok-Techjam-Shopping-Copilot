from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item import Candidate
from src.retrieval import Catalog
from src.retrieval.evaluation.first_turn import evaluate_recall
from src.retrieval.evaluation.multiturn import evaluate_multiturn_recall
from src.retrieval.experiments.hybrid_comparison import compare_hybrid_methods
from src.retrieval.experiments.text_ablation import run_text_ablation
from src.retrieval.experiments.visualize_results import render_html


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


class TurnSequenceRetriever:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.calls = 0

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> list[Candidate]:
        del query, state, intent
        self.calls += 1
        identifiers = (
            ["OTHER", "TARGET-1"]
            if self.calls == 1
            else ["TARGET-1", "OTHER"]
        )
        return [
            Candidate(
                item=self.catalog[parent_asin],
                retrieval_score=float(len(identifiers) - rank),
                retrieval_rank=rank,
            )
            for rank, parent_asin in enumerate(identifiers[:k], start=1)
        ]


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

    def test_multiturn_recall_records_each_turn_and_cumulative_hit(self) -> None:
        sample = {
            **_sample("multi", "browsing", "TARGET-1"),
            "intent_card": {
                "target_category": "shirt",
                "hard_constraints": ["soft cotton"],
                "soft_preferences": ["black"],
            },
            "behavior": {"scenario_type": "browsing"},
        }
        result = evaluate_multiturn_recall(
            TurnSequenceRetriever(self.catalog),
            self.catalog,
            [sample],
            ks=(1, 2, 10, 100),
            max_turns=2,
            result_top_n=1,
            stop_k=1,
            continue_after_hit=True,
            ask_policy=lambda state, candidates: "feature",
        )

        self.assertEqual(len(result["sessions"][0]["turns"]), 2)
        self.assertEqual(result["turn_metrics"][0]["strict_recall"]["recall_at_1"], 0.0)
        self.assertEqual(result["turn_metrics"][1]["strict_recall"]["recall_at_1"], 1.0)
        self.assertEqual(result["turn_metrics"][0]["officially_active_count"], 1)
        self.assertEqual(result["turn_metrics"][0]["remaining_unhit_at_1"], 1)
        self.assertEqual(result["turn_metrics"][1]["remaining_unhit_at_1"], 0)
        self.assertEqual(result["turn_metrics"][0]["session_hit_rate_at_2"], 1.0)
        self.assertEqual(result["overall"]["session_hit_rate_at_1"], 1.0)
        self.assertEqual(result["sessions"][0]["turns"][1]["target_rank"], 1)
        self.assertEqual(len(result["sessions"][0]["turns"][0]["top_results"]), 1)
        self.assertFalse(result["sessions"][0]["turns"][1]["post_hit_counterfactual"])

        report = render_html(result)
        self.assertIn("总体：逐轮 Retrieval Hit@100", report)
        self.assertIn("命中 1/1（100.0%） · 未命中 0", report)
        self.assertIn("四场景 Retrieval 整体效果", report)
        self.assertIn("单轮 Conditional Hit@100", report)
        self.assertIn("展开严格 Recall 与 Top50/100 诊断", report)
        self.assertIn("固定实验流程", report)
        self.assertNotIn('"sample_id"', report)


if __name__ == "__main__":
    unittest.main()
