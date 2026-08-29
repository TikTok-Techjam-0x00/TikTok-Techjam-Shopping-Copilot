from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.item import Candidate, Item
from src.reranking import SimpleReranker
from src.reranking.replay.evaluator import (
    RetrievalOrderRanker,
    evaluate_replay,
    load_replay_dataset,
)
from src.reranking.replay.recorder import collect_replay_dataset


class RankAllTest(unittest.TestCase):
    def test_rank_all_does_not_expand_production_top10_contract(self) -> None:
        candidates = [
            Candidate(
                item=Item(parent_asin=f"P{index:02d}", title=f"Running shoe {index}"),
                retrieval_score=float(20 - index),
                retrieval_rank=index,
            )
            for index in range(1, 13)
        ]
        state = {
            "intent": "browsing",
            "hard_constraint": {},
            "soft_constraint": {},
            "no_prefernce": [],
            "rejected_values": {},
            "user_profile": {},
            "user_message": "running shoes",
        }
        reranker = SimpleReranker()
        self.assertEqual(len(reranker.rerank(state, candidates)), 10)
        self.assertEqual(len(reranker.rank_all(state, candidates)), 12)


class ReplayEvaluatorIntegrationTest(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path]:
        catalog_path = root / "catalog.jsonl"
        products = [
            {
                "parent_asin": "TARGET",
                "title": "Black lightweight running shoes",
                "features": ["comfortable mesh", "water resistant"],
                "description": ["training shoes for daily running"],
                "price": 69.0,
                "categories": ["Clothing, Shoes & Jewelry", "Women", "Running Shoes"],
                "details": {"Color": "Black", "Material": "Mesh"},
                "average_rating": 4.7,
                "rating_number": 120,
                "store": "Example",
            },
            {
                "parent_asin": "OTHER",
                "title": "White leather formal shoes",
                "features": ["formal style"],
                "description": ["office footwear"],
                "price": 130.0,
                "categories": ["Clothing, Shoes & Jewelry", "Men", "Formal Shoes"],
                "details": {"Color": "White", "Material": "Leather"},
                "average_rating": 4.0,
                "rating_number": 10,
                "store": "Example",
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        dataset_path = root / "public_set.jsonl"
        samples = [
            {
                "sample_id": "public_test_buying",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": "TARGET"},
                "user_profile": {"preference_tags": ["comfort"]},
            },
            {
                "sample_id": "public_test_override",
                "scenario_type": "intent_override",
                "ground_truth": {"parent_asin": "TARGET"},
                "user_profile": {"preference_tags": ["running"]},
                "intent_card": {
                    "target_category": "running shoes",
                    "hard_constraints": ["material: mesh"],
                    "soft_preferences": ["color: black"],
                },
                "behavior": {
                    "scenario_type": "intent_override",
                    "override": {
                        "turn": 3,
                        "old_value": "color: white",
                        "new_value": "material: mesh",
                        "message": "Actually, ignore white. I need material: mesh.",
                    },
                },
            },
        ]
        dataset_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )
        return catalog_path, dataset_path

    def test_record_versioned_cases_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path, dataset_path = self._write_inputs(root)
            replay_directory = collect_replay_dataset(
                catalog_path=catalog_path,
                dataset_path=dataset_path,
                output_root=root / "replays",
                run_id="test-dataset-v1",
                max_turns=3,
                command=["unit-test", "record"],
            )
            manifest, cases, labels = load_replay_dataset(replay_directory)
            self.assertEqual(manifest["run_id"], "test-dataset-v1")
            self.assertEqual(manifest["counts"]["cases"], 6)
            self.assertEqual(
                manifest["generation_policy"]["runtime_reranker_execution"],
                "retrieval_order_passthrough",
            )
            self.assertIn("component_versions", manifest["generation_provenance"])
            self.assertIn("commit", manifest["generation_provenance"]["git"])
            self.assertNotEqual(manifest["generation_provenance"]["git"]["commit"], "unknown")
            self.assertEqual(
                manifest["generation_provenance"]["inputs"]["dataset"]["role"],
                "public_evaluator_set",
            )
            self.assertIn(
                "last_committed_change",
                manifest["generation_provenance"]["component_versions"]["state"],
            )
            self.assertEqual(len(cases), len(labels))

            override_cases = [
                case for case in cases if case.sample_id == "public_test_override"
            ]
            self.assertEqual([case.scorable for case in override_cases], [False, False, True])
            self.assertNotIn("target_parent_asin", override_cases[0].to_dict())

            result_directory = evaluate_replay(
                replay_directory,
                catalog_path=catalog_path,
                experiments={"s1_rule_fuzzy": SimpleReranker()},
                experiment_id="RR-999",
                command=["unit-test", "evaluate"],
            )
            report = json.loads((result_directory / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["dataset_run_id"], "test-dataset-v1")
            self.assertEqual(report["experiment_id"], "RR-999")
            self.assertEqual(report["evaluation_run_id"], "RR-999")
            self.assertEqual(result_directory.name, "RR-999")
            self.assertGreaterEqual(report["total_elapsed_seconds"], 0.0)
            self.assertIn("dataset_git_commit", report)
            self.assertIn("evaluation_git_commit", report)
            self.assertEqual(set(report["experiments"]), {"s1_rule_fuzzy"})
            metadata = report["experiments"]["s1_rule_fuzzy"]["metadata"]
            self.assertIn("hard_constraint_strategy", metadata)
            self.assertIn("score_fusion", metadata)
            self.assertTrue((result_directory / "report.md").is_file())
            self.assertTrue((result_directory / "case_results.jsonl.gz").is_file())

    def test_one_experiment_id_cannot_mix_configurations(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate_replay(
                ".",
                catalog_path="data/catalog.jsonl",
                experiments={
                    "retrieval_order": RetrievalOrderRanker(),
                    "s1_rule_fuzzy": SimpleReranker(),
                },
                experiment_id="RR-998",
            )


if __name__ == "__main__":
    unittest.main()
