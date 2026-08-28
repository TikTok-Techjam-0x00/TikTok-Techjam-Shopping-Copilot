"""Smoke and isolation tests for the standalone 3B benchmark variants."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import ModuleType

from . import three_b


EXPERIMENT_DIR = Path(__file__).with_name("experiments")
VARIANT_NAMES = (
    "baseline",
    "profile_boost",
    "turn_boost",
    "profile_turn",
    "legacy_base",
    "diversity_12",
    "diversity_24",
    "diversity_38",
    "unweighted_coverage",
    "cardinality_penalty",
    "historical_composite",
)

CURRENT_BASE = {
    "category": 90.0,
    "use_case": 60.0,
    "feature": 65.0,
    "size": 60.0,
    "material": 65.0,
    "budget": 55.0,
    "style": 60.0,
    "color": 60.0,
    "brand": 35.0,
    "other": 5.0,
}
LEGACY_BASE = {
    "category": 90.0,
    "use_case": 70.0,
    "feature": 68.0,
    "size": 66.0,
    "material": 64.0,
    "budget": 60.0,
    "style": 58.0,
    "color": 52.0,
    "brand": 45.0,
    "other": 5.0,
}


def _load_variant(name: str) -> ModuleType:
    """Load a replacement file with the same package context as production 3B."""
    path = EXPERIMENT_DIR / f"three_b_exp_{name}.py"
    module_name = f"src.dialogue._three_b_experiment_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment variant: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _state(turn: int = 1, profile_tags: list[str] | None = None) -> dict:
    return {
        "turn": turn,
        "user_profile": {"preference_tags": profile_tags or []},
        "hard_constraint": {"category": "shoes"},
        "soft_constraint": {},
        "no_prefernce": [],
        "asked_attributes": [],
    }


def _candidates() -> list[dict]:
    return [
        {
            "item": {
                "category": "shoes",
                "features": ["waterproof"],
                "material": "leather",
                "color": "black",
                "price": 49.0,
            }
        },
        {
            "item": {
                "category": "shoes",
                "features": ["breathable"],
                "material": "cotton",
                "color": "white",
                "price": 79.0,
            }
        },
    ]


class ThreeBExperimentVariantsTest(unittest.TestCase):
    def test_all_variants_import_and_keep_public_interface(self) -> None:
        public_functions = (
            "choose_ask_attribute",
            "build_question",
            "decide_ask",
            "record_asked_attribute",
        )
        for name in VARIANT_NAMES:
            with self.subTest(name=name):
                module = _load_variant(name)
                for function_name in public_functions:
                    self.assertEqual(
                        inspect.signature(getattr(module, function_name)),
                        inspect.signature(getattr(three_b, function_name)),
                    )
                decision = module.decide_ask(_state(), _candidates())
                self.assertEqual(set(decision), {"ask_attribute", "message"})
                self.assertIn(decision["ask_attribute"], module.ATTRIBUTES)
                self.assertIsInstance(decision["message"], str)
                selector_decision = module.AskAttributeSelector().decide(
                    _state(), _candidates()
                )
                self.assertEqual(selector_decision, decision)

    def test_baseline_body_is_exact_production_copy(self) -> None:
        production = Path(three_b.__file__).read_text(encoding="utf-8")
        baseline = (
            EXPERIMENT_DIR / "three_b_exp_baseline.py"
        ).read_text(encoding="utf-8")
        body_start = baseline.index('"""3B:')
        self.assertEqual(baseline[body_start:], production)

        module = _load_variant("baseline")
        for state in (_state(), _state(turn=10), {"turn": 1}):
            self.assertEqual(
                module.decide_ask(state, _candidates()),
                three_b.decide_ask(state, _candidates()),
            )

    def test_headers_document_every_variant(self) -> None:
        required = (
            "# 3B EXPERIMENT VARIANT",
            "# Baseline:",
            "# Differences from current baseline:",
            "# Variables changed:",
            "# Variables intentionally kept unchanged:",
            "# Purpose:",
        )
        for name in VARIANT_NAMES:
            with self.subTest(name=name):
                source = (
                    EXPERIMENT_DIR / f"three_b_exp_{name}.py"
                ).read_text(encoding="utf-8")
                for marker in required:
                    self.assertIn(marker, source[:2500])

    def test_variant_constants_match_experiment_matrix(self) -> None:
        for name in (
            "baseline",
            "profile_boost",
            "turn_boost",
            "profile_turn",
            "diversity_12",
            "diversity_24",
            "diversity_38",
            "unweighted_coverage",
            "cardinality_penalty",
        ):
            expected_base = LEGACY_BASE if name == "legacy_base" else CURRENT_BASE
            self.assertEqual(_load_variant(name).BASE_PRIORITY, expected_base)

        self.assertEqual(_load_variant("legacy_base").BASE_PRIORITY, LEGACY_BASE)
        self.assertEqual(
            _load_variant("historical_composite").BASE_PRIORITY, LEGACY_BASE
        )
        for name, coefficient in (
            ("baseline", 18.0),
            ("diversity_12", 12.0),
            ("diversity_24", 24.0),
            ("diversity_38", 38.0),
            ("historical_composite", 38.0),
        ):
            self.assertEqual(
                _load_variant(name).DIVERSITY_COEFFICIENT, coefficient
            )

    def test_historical_profile_and_turn_policies_are_active(self) -> None:
        profile = _load_variant("profile_boost")
        boosts = profile._profile_boosts(
            _state(profile_tags=["comfort", "durability", "fit"])
        )
        self.assertEqual(boosts["feature"], 24.0)
        self.assertEqual(boosts["size"], 12.0)
        self.assertEqual(
            profile.choose_ask_attribute(
                _state(profile_tags=["fit"]), []
            )[0],
            "size",
        )

        baseline = _load_variant("baseline")
        turn = _load_variant("turn_boost")
        empty_candidates: list[dict] = []
        self.assertEqual(
            baseline.choose_ask_attribute(_state(turn=1), empty_candidates)[0],
            "feature",
        )
        self.assertEqual(
            turn.choose_ask_attribute(_state(turn=1), empty_candidates)[0],
            "use_case",
        )

    def test_coverage_and_cardinality_variants_change_only_dynamic_signal(self) -> None:
        baseline = _load_variant("baseline")
        ordinary = _load_variant("unweighted_coverage")
        sparse = [
            {"item": {"color": "red"}},
            {"item": {}},
            {"item": {}},
            {"item": {"color": "blue"}},
        ]
        baseline_signal = baseline._candidate_diversity_signal(sparse, "color")[0]
        ordinary_signal = ordinary._candidate_diversity_signal(sparse, "color")[0]
        self.assertNotAlmostEqual(baseline_signal, ordinary_signal)

        cardinality = _load_variant("cardinality_penalty")
        many_values = [
            {"item": {"color": f"color-{index}"}} for index in range(6)
        ]
        no_penalty = baseline._candidate_diversity_signal(
            many_values, "color"
        )[0]
        with_penalty = cardinality._candidate_diversity_signal(
            many_values, "color"
        )[0]
        self.assertAlmostEqual(with_penalty, no_penalty * 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
