"""Run reproducible BM25 product-text ablations on the public set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import load_jsonl

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.catalog import Catalog
from src.retrieval.evaluate import DEFAULT_KS, evaluate_recall
from src.retrieval.text import TEXT_CONFIGS


def run_text_ablation(
    catalog: Catalog,
    samples: Sequence[dict[str, Any]],
    *,
    versions: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> list[dict[str, Any]]:
    """Build and evaluate each version independently using the same samples."""
    results: list[dict[str, Any]] = []
    for version in versions:
        config = TEXT_CONFIGS[version]
        started = time.perf_counter()
        retriever = BM25Retriever(catalog, text_version=config)
        index_build_seconds = time.perf_counter() - started
        try:
            metrics = evaluate_recall(retriever, catalog, samples, ks=ks)
        finally:
            retriever.close()
        results.append(
            {
                "text_version": config.name,
                "fields": list(config.fields),
                "description": config.description,
                "index_build_seconds": round(index_build_seconds, 3),
                **metrics,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare BM25 product text versions using Recall@K."
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
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=tuple(TEXT_CONFIGS),
        default=list(TEXT_CONFIGS),
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "bm25_text_ablation.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    started = time.perf_counter()
    catalog = Catalog.load(args.catalog)
    catalog_load_seconds = time.perf_counter() - started
    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    result = {
        "experiment": "bm25_text_ablation_v1",
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "catalog_load_seconds": round(catalog_load_seconds, 3),
        "sample_count": len(samples),
        "results": run_text_ablation(
            catalog,
            samples,
            versions=args.versions,
            ks=args.ks,
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
