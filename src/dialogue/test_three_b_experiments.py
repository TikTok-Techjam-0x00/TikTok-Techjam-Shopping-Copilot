"""Smoke and isolation tests for the standalone 3B benchmark variants."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import ModuleType

from ..attribute import AttributeName, AttributeValue
from ..item import Candidate, Item
from ..state import ShoppingState
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
    "diversity_38_cardinality",
    "diversity_38_legacy_base",
    "unweighted_coverage",
    "cardinality_penalty",
    "historical_composite",
    "historical_minus_legacy_base",
    "historical_minus_profile",
    "historical_minus_turn",
    "historical_minus_diversity_38",
    "historical_minus_ordinary_coverage",
    "historical_minus_cardinality",
    "historical_override_no_profile",
    "historical_override_reask",
    "historical_override_adaptive",
    "historical_override_phase_no_profile",
    "scenario_adaptive",
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

    def test_baseline_snapshot_remains_original_control(self) -> None:
        production = Path(three_b.__file__).read_text(encoding="utf-8")
        baseline = (
            EXPERIMENT_DIR / "three_b_exp_baseline.py"
        ).read_text(encoding="utf-8")
        body_start = baseline.index('"""3B:')
        self.assertNotEqual(baseline[body_start:], production)

        module = _load_variant("baseline")
        self.assertEqual(module.BASE_PRIORITY, CURRENT_BASE)
        self.assertEqual(module.DIVERSITY_COEFFICIENT, 18.0)

    def test_production_body_matches_promoted_historical_composite(self) -> None:
        production = Path(three_b.__file__).read_text(encoding="utf-8")
        composite = (
            EXPERIMENT_DIR / "three_b_exp_historical_composite.py"
        ).read_text(encoding="utf-8")
        body_start = composite.index('"""3B:')
        self.assertEqual(composite[body_start:], production)

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
            "diversity_38_cardinality",
            "unweighted_coverage",
            "cardinality_penalty",
        ):
            expected_base = LEGACY_BASE if name == "legacy_base" else CURRENT_BASE
            self.assertEqual(_load_variant(name).BASE_PRIORITY, expected_base)

        self.assertEqual(_load_variant("legacy_base").BASE_PRIORITY, LEGACY_BASE)
        self.assertEqual(
            _load_variant("diversity_38_legacy_base").BASE_PRIORITY,
            LEGACY_BASE,
        )
        self.assertEqual(
            _load_variant("historical_composite").BASE_PRIORITY, LEGACY_BASE
        )
        self.assertEqual(
            _load_variant("historical_minus_legacy_base").BASE_PRIORITY,
            CURRENT_BASE,
        )
        for name in (
            "historical_minus_profile",
            "historical_minus_turn",
            "historical_minus_diversity_38",
            "historical_minus_ordinary_coverage",
            "historical_minus_cardinality",
        ):
            self.assertEqual(_load_variant(name).BASE_PRIORITY, LEGACY_BASE)
        for name, coefficient in (
            ("baseline", 18.0),
            ("diversity_12", 12.0),
            ("diversity_24", 24.0),
            ("diversity_38", 38.0),
            ("diversity_38_cardinality", 38.0),
            ("diversity_38_legacy_base", 38.0),
            ("historical_composite", 38.0),
            ("historical_minus_legacy_base", 38.0),
            ("historical_minus_profile", 38.0),
            ("historical_minus_turn", 38.0),
            ("historical_minus_diversity_38", 18.0),
            ("historical_minus_ordinary_coverage", 38.0),
            ("historical_minus_cardinality", 38.0),
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

        diversity_38 = _load_variant("diversity_38")
        combined = _load_variant("diversity_38_cardinality")
        no_penalty_38 = diversity_38._candidate_diversity_signal(
            many_values, "color"
        )[0]
        with_penalty_38 = combined._candidate_diversity_signal(
            many_values, "color"
        )[0]
        self.assertAlmostEqual(with_penalty_38, no_penalty_38 * 5.0 / 6.0)

    def test_override_variants_use_only_module_two_runtime_signal(self) -> None:
        state = _state(turn=3, profile_tags=["fit"])
        state["override_detected"] = True
        state["asked_attributes"] = ["use_case"]

        composite = _load_variant("historical_composite")
        no_profile = _load_variant("historical_override_no_profile")
        reask = _load_variant("historical_override_reask")
        adaptive = _load_variant("historical_override_adaptive")

        self.assertEqual(
            composite.choose_ask_attribute(state, [])[0],
            "size",
        )
        self.assertEqual(
            no_profile.choose_ask_attribute(state, [])[0],
            "feature",
        )
        self.assertEqual(
            reask.choose_ask_attribute(state, [])[0],
            "size",
        )
        self.assertEqual(
            adaptive.choose_ask_attribute(state, [])[0],
            "use_case",
        )

        later_state = _state(turn=3, profile_tags=["fit"])
        later_state["override_detected"] = False
        later_state["history"] = [
            "I'm looking for running shoes.",
            "Actually, change that to hiking boots.",
            "For that, what matters is: waterproof.",
        ]
        phase_policy = _load_variant("historical_override_phase_no_profile")
        self.assertEqual(
            composite.choose_ask_attribute(later_state, [])[0],
            "size",
        )
        self.assertEqual(
            phase_policy.choose_ask_attribute(later_state, [])[0],
            "use_case",
        )


class ScenarioAdaptiveExperimentTest(unittest.TestCase):
    """Behavioral coverage for the composite scenario-adaptive policy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_variant("scenario_adaptive")

    def test_mapping_object_candidate_and_public_contract(self) -> None:
        mapping_state = _state()
        mapping_candidate = {
            "item": {"category": "shoes", "material": "cotton"}
        }
        mapping_decision = self.module.decide_ask(
            mapping_state, [mapping_candidate]
        )
        self.assertEqual(set(mapping_decision), {"ask_attribute", "message"})

        object_state = ShoppingState(
            session_id="scenario-object",
            turn=1,
            hard_constraint={
                AttributeName.CATEGORY: AttributeValue(values=["shoes"])
            },
        )
        object_candidate = Candidate(
            item=Item(
                parent_asin="scenario-item",
                categories=["shoes"],
                details={"Material": "cotton"},
            )
        )
        object_decision = self.module.decide_ask(
            object_state, [object_candidate]
        )
        self.assertIn(object_decision["ask_attribute"], self.module.ATTRIBUTES)

        mapping_state["turn"] = 10
        self.assertEqual(
            self.module.decide_ask(mapping_state, [mapping_candidate]),
            {"ask_attribute": None, "message": ""},
        )
        unknown_category_state = _state()
        unknown_category_state["hard_constraint"] = {}
        self.assertEqual(
            self.module.choose_ask_attribute(
                unknown_category_state, [mapping_candidate]
            )[0],
            "category",
        )

    def test_fragmented_feature_has_no_answer_yield_or_options(self) -> None:
        candidates = [
            {
                "item": {
                    "features": [f"unique capability {index}"],
                    "material": "cotton" if index % 2 else "leather",
                }
            }
            for index in range(30)
        ]
        feature = self.module._attribute_signals(candidates, "feature")
        self.assertEqual(feature["repeatability"], 0.0)
        self.assertEqual(feature["answer_yield"], 0.0)
        self.assertEqual(feature["options"], [])
        self.assertEqual(
            self.module.choose_ask_attribute(_state(), candidates)[0],
            "material",
        )

    def test_buying_uses_diversity_inside_safe_conditional_pool(self) -> None:
        cotton = [
            {
                "item": {
                    "material": "cotton",
                    "color": "black",
                    "size": "small" if index % 2 else "large",
                }
            }
            for index in range(15)
        ]
        leather = [
            {
                "item": {
                    "material": "leather",
                    "color": f"catalog-color-{index}",
                }
            }
            for index in range(25)
        ]
        candidates = cotton + leather
        state = _state()
        state.update(
            {
                "intent": "buying",
                "hard_constraint": {
                    "category": "shoes",
                    "material": "cotton",
                },
            }
        )

        conditional = self.module._conditional_items_for_buying(
            state, candidates
        )
        self.assertEqual(len(conditional), 15)
        self.assertGreater(
            self.module._attribute_signals(candidates, "color")["gini_top100"],
            self.module._attribute_signals(conditional, "color")["gini_top100"],
        )
        self.assertEqual(
            self.module.choose_ask_attribute(state, candidates)[0],
            "size",
        )

    def test_conditional_pool_safely_falls_back_when_too_small(self) -> None:
        state = _state()
        state.update(
            {
                "intent": "buying",
                "hard_constraint": {
                    "category": "shoes",
                    "material": "cotton",
                },
            }
        )
        candidates = [
            {"item": {}},
            {"item": {"material": "cotton", "color": "black"}},
            {"item": {"material": "leather", "color": "white"}},
        ]
        self.assertEqual(
            self.module._conditional_items_for_buying(state, candidates),
            candidates,
        )
        self.assertIn(
            self.module.choose_ask_attribute(state, candidates)[0],
            self.module.ATTRIBUTES,
        )

    def test_boundary_mode_prefers_high_answer_yield(self) -> None:
        state = _state()
        state["boundary_detected"] = True
        candidates = [
            {
                "item": {
                    "material": f"one-off-material-{index}",
                    "brand": "alpha" if index % 2 else "beta",
                }
            }
            for index in range(24)
        ]
        self.assertEqual(self.module._policy_mode(state), "BOUNDARY_RECOVER")
        self.assertEqual(
            self.module.choose_ask_attribute(state, candidates)[0],
            "brand",
        )

    def test_override_reopens_attributes_and_resets_asked_epoch(self) -> None:
        state = _state()
        state.update(
            {
                "override_detected": True,
                "asked_attributes": ["size", "brand"],
            }
        )
        candidates = [
            {
                "item": {
                    "size": "small" if index % 2 else "large",
                    "brand": "same-brand",
                }
            }
            for index in range(20)
        ]
        attribute, _ = self.module.choose_ask_attribute(state, candidates)
        self.assertEqual(attribute, "size")
        self.module.record_asked_attribute(state, attribute)
        self.assertEqual(state["asked_attributes"], ["size"])

        object_state = ShoppingState(
            session_id="override-object",
            turn=3,
            hard_constraint={
                AttributeName.CATEGORY: AttributeValue(values=["shoes"])
            },
            asked_attributes=["size", "brand"],
            override_detected=True,
        )
        self.module.record_asked_attribute(object_state, "color")
        self.assertEqual(object_state.asked_attributes, ["color"])

    def test_equal_scores_follow_official_attribute_order(self) -> None:
        self.assertEqual(
            self.module.choose_ask_attribute(_state(), [])[0],
            "material",
        )

    def test_other_is_fallback_only(self) -> None:
        state = _state()
        state["asked_attributes"] = [
            attribute
            for attribute in self.module.ATTRIBUTES
            if attribute not in {"category", "other"}
        ]
        self.assertEqual(
            self.module.choose_ask_attribute(state, [])[0],
            "other",
        )
        state["asked_attributes"].append("other")
        self.assertEqual(
            self.module.choose_ask_attribute(state, []),
            (None, []),
        )


if __name__ == "__main__":
    unittest.main()
