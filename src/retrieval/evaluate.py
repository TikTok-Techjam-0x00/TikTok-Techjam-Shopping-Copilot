"""Evaluate first-turn Retrieval recall on the released public sessions.

This module does not modify or replace the official end-to-end evaluator. It
reuses the official simulator helpers to materialize each public session's first
user message, retrieves up to max(K), and checks the exact target parent_asin.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


# Support both `python -m src.retrieval.evaluate` and VSCode's direct
# "Run Python File" action, which otherwise only adds this file's directory to
# sys.path instead of the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)

from src.item import Candidate
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.catalog import Catalog
from src.retrieval.retriever import RetrievalStrategy
from src.retrieval.text import DEFAULT_TEXT_VERSION, TEXT_CONFIGS


DEFAULT_KS = (10, 50, 100)


def _validated_ks(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("at least one recall K is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("recall K values must be positive integers")
    return tuple(sorted(set(values)))


def _target_asin(sample: Mapping[str, Any]) -> str:
    ground_truth = sample.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("sample.ground_truth must be an object")
    target = str(ground_truth.get("parent_asin") or "").strip()
    if not target:
        raise ValueError("sample.ground_truth.parent_asin must not be empty")
    return target


def _official_initial_query(
    sample: Mapping[str, Any],
    catalog: Catalog,
) -> str:
    """Materialize the same first user message as the official evaluator."""
    target = _target_asin(sample)
    product = catalog.get(target)
    if product is None:
        raise ValueError(f"target parent_asin {target!r} is missing from the catalog")
    product_dict = product.to_dict()
    card, behavior = materialize_hidden_fields(dict(sample), {target: product_dict})
    effective_sample = {
        **sample,
        "intent_card": card,
        "behavior": behavior,
    }
    return initial_message(
        effective_sample,
        coarse_category(product.categories),
        set(),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _recall_summary(ranks: Sequence[int | None], ks: Sequence[int]) -> dict[str, Any]:
    sample_count = len(ranks)
    result: dict[str, Any] = {"sample_count": sample_count}
    for k in ks:
        hits = sum(rank is not None and rank <= k for rank in ranks)
        result[f"hits_at_{k}"] = hits
        result[f"recall_at_{k}"] = round(hits / sample_count, 6) if sample_count else 0.0
    return result


def evaluate_recall(
    retriever: RetrievalStrategy,
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    include_sessions: bool = False,
) -> dict[str, Any]:
    """Measure exact target recall for official first-turn public queries.

    The target ASIN is used only to materialize the official simulated message and
    to judge the returned rank. It is never supplied to ``retriever.retrieve``.
    """
    recall_ks = _validated_ks(ks)
    max_k = recall_ks[-1]
    ranks: list[int | None] = []
    query_latencies: list[float] = []
    grouped_ranks: dict[str, list[int | None]] = defaultdict(list)
    sessions: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample {index} must be an object")
        target = _target_asin(sample)
        query = _official_initial_query(sample, catalog)

        started = time.perf_counter()
        candidates = retriever.retrieve(query=query, state=None, intent=None, k=max_k)
        query_latencies.append(time.perf_counter() - started)

        target_rank: int | None = None
        seen: set[str] = set()
        for rank, candidate in enumerate(candidates[:max_k], start=1):
            if not isinstance(candidate, Candidate):
                raise TypeError("retriever must return shared Candidate objects")
            if candidate.parent_asin in seen:
                raise ValueError("retriever returned duplicate parent_asin values")
            seen.add(candidate.parent_asin)
            if candidate.parent_asin == target and target_rank is None:
                target_rank = rank

        scenario = str(sample.get("scenario_type") or "unknown")
        ranks.append(target_rank)
        grouped_ranks[scenario].append(target_rank)
        if include_sessions:
            sessions.append(
                {
                    "sample_id": str(sample.get("sample_id") or index),
                    "scenario_type": scenario,
                    "target_parent_asin": target,
                    "query": query,
                    "target_rank": target_rank,
                }
            )

    latency_ms = [value * 1000.0 for value in query_latencies]
    result: dict[str, Any] = {
        "query_mode": "official_first_turn",
        "ks": list(recall_ks),
        "overall": _recall_summary(ranks, recall_ks),
        "scenario_metrics": {
            scenario: _recall_summary(scenario_ranks, recall_ks)
            for scenario, scenario_ranks in sorted(grouped_ranks.items())
        },
        "query_latency_ms": {
            "mean": round(statistics.fmean(latency_ms), 3) if latency_ms else 0.0,
            "p50": round(_percentile(latency_ms, 0.50), 3),
            "p95": round(_percentile(latency_ms, 0.95), 3),
            "max": round(max(latency_ms), 3) if latency_ms else 0.0,
        },
    }
    if include_sessions:
        result["sessions"] = sessions
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure BM25 Retrieval Recall@K on official first-turn public queries."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "catalog.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "public_set.jsonl",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--experiment", default="bm25_v0")
    parser.add_argument(
        "--text-version",
        choices=tuple(TEXT_CONFIGS),
        default=DEFAULT_TEXT_VERSION,
        help="Versioned product fields used to build the BM25 index.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--include-sessions",
        action="store_true",
        help="Include query and target rank for each public session in JSON output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    started = time.perf_counter()
    catalog = Catalog.load(args.catalog)
    catalog_load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    retriever = BM25Retriever(catalog, text_version=args.text_version)
    index_build_seconds = time.perf_counter() - started
    try:
        samples = load_jsonl(args.dataset)
        if args.limit is not None:
            samples = samples[: args.limit]
        result = evaluate_recall(
            retriever,
            catalog,
            samples,
            ks=args.ks,
            include_sessions=args.include_sessions,
        )
    finally:
        retriever.close()

    result = {
        "experiment": args.experiment,
        "text_version": args.text_version,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "catalog_load_seconds": round(catalog_load_seconds, 3),
        "index_build_seconds": round(index_build_seconds, 3),
        **result,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
