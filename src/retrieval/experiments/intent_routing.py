"""Compare the default BM25 strategy with Buying/Browsing text routing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import load_jsonl

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.catalog import Catalog
from src.retrieval.evaluation.multiturn import evaluate_multiturn_recall
from src.retrieval.routing import IntentRoutedRetriever, IntentRoutingConfig
from src.retrieval.text import DEFAULT_TEXT_VERSION, TEXT_CONFIGS


def _metric_delta(
    baseline: Mapping[str, Any],
    routed: Mapping[str, Any],
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for k in (10, 50, 100):
        for prefix in ("session_hits_at", "session_hit_rate_at", "mttc_at"):
            key = f"{prefix}_{k}"
            before = baseline.get(key)
            after = routed.get(key)
            result[key] = (
                round(float(after) - float(before), 6)
                if before is not None and after is not None
                else None
            )
    return result


def compare_intent_routing(
    baseline_retriever: BM25Retriever,
    routed_retriever: IntentRoutedRetriever,
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    *,
    max_turns: int = 10,
    result_top_n: int = 10,
) -> dict[str, Any]:
    baseline = evaluate_multiturn_recall(
        baseline_retriever,
        catalog,
        samples,
        max_turns=max_turns,
        result_top_n=result_top_n,
    )
    routed = evaluate_multiturn_recall(
        routed_retriever,
        catalog,
        samples,
        max_turns=max_turns,
        result_top_n=result_top_n,
    )
    scenarios: dict[str, Any] = {}
    scenario_no_regression = True
    for scenario in sorted(baseline["scenario_metrics"]):
        before = baseline["scenario_metrics"][scenario]["overall"]
        after = routed["scenario_metrics"][scenario]["overall"]
        no_regression = (
            int(after["session_hits_at_10"]) >= int(before["session_hits_at_10"])
            and int(after["session_hits_at_100"]) >= int(before["session_hits_at_100"])
        )
        scenario_no_regression = scenario_no_regression and no_regression
        scenarios[scenario] = {
            "baseline": before,
            "routed": after,
            "delta": _metric_delta(before, after),
            "no_hit_count_regression_at_10_and_100": no_regression,
        }

    before_overall = baseline["overall"]
    after_overall = routed["overall"]
    criteria = {
        "overall_hits_at_10_non_decreasing": (
            after_overall["session_hits_at_10"] >= before_overall["session_hits_at_10"]
        ),
        "overall_hits_at_100_non_decreasing": (
            after_overall["session_hits_at_100"] >= before_overall["session_hits_at_100"]
        ),
        "overall_mttc_at_10_non_increasing": (
            after_overall["mttc_at_10"] <= before_overall["mttc_at_10"]
        ),
        "every_scenario_hits_at_10_and_100_non_decreasing": scenario_no_regression,
    }
    return {
        "experiment": "bm25_intent_routing_warm_start_v2",
        "reliability_gate": {
            "passed": all(criteria.values()),
            "criteria": criteria,
            "decision_rule": (
                "Promote only if overall Top10/Top100 hit counts do not decrease, "
                "Top10 MTTC does not increase, and no scenario loses Top10 or Top100 hits."
            ),
        },
        "comparison": {
            "overall": {
                "baseline": before_overall,
                "routed": after_overall,
                "delta": _metric_delta(before_overall, after_overall),
            },
            "scenarios": scenarios,
        },
        "baseline": baseline,
        "routed": routed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BM25 Buying/Browsing routing.")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--buying-text-version", choices=tuple(TEXT_CONFIGS), default=DEFAULT_TEXT_VERSION)
    parser.add_argument("--browsing-text-version", choices=tuple(TEXT_CONFIGS), default="title_category_v1")
    parser.add_argument("--browsing-max-turn", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--result-top-n", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "retrieval" / "intent_routing_warm_start_v2.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    catalog = Catalog.load(args.catalog)
    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    buying = BM25Retriever(catalog, text_version=args.buying_text_version)
    browsing = BM25Retriever(catalog, text_version=args.browsing_text_version)
    routed = IntentRoutedRetriever(
        buying,
        browsing,
        config=IntentRoutingConfig(browsing_max_turn=args.browsing_max_turn),
    )
    try:
        result = compare_intent_routing(
            buying,
            routed,
            catalog,
            samples,
            max_turns=args.max_turns,
            result_top_n=args.result_top_n,
        )
    finally:
        buying.close()
        browsing.close()
    result = {
        **result,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "sample_count": len(samples),
        "buying_text_version": args.buying_text_version,
        "browsing_text_version": args.browsing_text_version,
        "browsing_max_turn": args.browsing_max_turn,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "reliability_gate": result["reliability_gate"],
                "comparison": result["comparison"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
