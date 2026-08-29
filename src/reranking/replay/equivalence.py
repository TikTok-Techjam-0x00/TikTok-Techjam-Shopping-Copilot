"""Verify that S1-fast preserves every observable full-ranking output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item import RankedCandidate
from src.retrieval import Catalog
from src.reranking import FastRuleFuzzyScorer, RuleFuzzyScorer, SimpleReranker
from src.reranking.replay.evaluator import (
    _restore_candidates,
    _validate_ranking,
    load_replay_dataset,
)
from src.reranking.replay.provenance import collect_provenance, sha256_file


def _ranking_signature(ranking: Sequence[RankedCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "parent_asin": candidate.parent_asin,
            "rerank_rank": candidate.rerank_rank,
            "rerank_score": candidate.rerank_score,
            "matched": list(candidate.matched),
            "violation": list(candidate.violation),
        }
        for candidate in ranking
    ]


def verify_s1_fast_equivalence(
    dataset_directory: str | Path,
    *,
    catalog_path: str | Path,
    limit: int | None = None,
    progress_every: int = 0,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Compare uncached S1 and S1-fast across the complete ranking contract."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must not be negative")
    dataset_directory = Path(dataset_directory).resolve()
    catalog_path = Path(catalog_path).resolve()
    manifest, cases, _ = load_replay_dataset(dataset_directory)
    if limit is not None:
        cases = cases[:limit]
    recorded_catalog = manifest["generation_provenance"]["inputs"]["catalog"]
    if sha256_file(catalog_path) != recorded_catalog["sha256"]:
        raise ValueError("catalog checksum differs from the replay dataset")

    catalog = Catalog.load(catalog_path)
    baseline = SimpleReranker(relevance_scorer=RuleFuzzyScorer())
    fast_scorer = FastRuleFuzzyScorer()
    optimized = SimpleReranker(relevance_scorer=fast_scorer)
    baseline_seconds = 0.0
    optimized_seconds = 0.0
    compared_candidates = 0
    started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        candidates = _restore_candidates(case, catalog)
        baseline_started = time.perf_counter()
        expected = baseline.rank_all(case.shopping_state, candidates)
        baseline_seconds += time.perf_counter() - baseline_started
        optimized_started = time.perf_counter()
        actual = optimized.rank_all(case.shopping_state, candidates)
        optimized_seconds += time.perf_counter() - optimized_started
        _validate_ranking(case, expected)
        _validate_ranking(case, actual)
        expected_signature = _ranking_signature(expected)
        actual_signature = _ranking_signature(actual)
        if actual_signature != expected_signature:
            first_difference = next(
                (
                    position
                    for position, (left, right) in enumerate(
                        zip(expected_signature, actual_signature, strict=False),
                        start=1,
                    )
                    if left != right
                ),
                min(len(expected_signature), len(actual_signature)) + 1,
            )
            raise AssertionError(
                f"{case.case_id}: S1-fast differs from baseline at rank "
                f"{first_difference}"
            )
        compared_candidates += len(expected_signature)
        if progress_every and (index % progress_every == 0 or index == len(cases)):
            print(
                json.dumps(
                    {
                        "equivalent_cases": index,
                        "total_cases": len(cases),
                        "compared_candidates": compared_candidates,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    provenance = collect_provenance(
        PROJECT_ROOT,
        catalog_path=catalog_path,
        dataset_path=dataset_directory / "cases.jsonl.gz",
        dataset_role="reranking_replay_cases",
        command=command,
    )
    return {
        "schema_version": "s1-fast-equivalence-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_run_id": manifest["run_id"],
        "dataset_manifest": str((dataset_directory / "manifest.json").resolve()),
        "equivalent": True,
        "compared_cases": len(cases),
        "compared_candidates": compared_candidates,
        "observable_fields": [
            "parent_asin",
            "rerank_rank",
            "rerank_score",
            "matched",
            "violation",
        ],
        "baseline_seconds": round(baseline_seconds, 3),
        "optimized_seconds": round(optimized_seconds, 3),
        "paired_elapsed_seconds": round(time.perf_counter() - started, 3),
        "paired_speedup": round(baseline_seconds / optimized_seconds, 6)
        if optimized_seconds
        else None,
        "fast_cache": fast_scorer.cache_info(),
        "verification_provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify exact S1-fast ranking equivalence.")
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_s1_fast_equivalence(
        args.dataset_directory,
        catalog_path=args.catalog,
        limit=args.limit,
        progress_every=args.progress_every,
        command=[sys.executable, "-m", "src.reranking.replay.equivalence", *(argv or sys.argv[1:])],
    )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {**report, "report_path": str(output)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
