"""Compare symmetric, query-typed, and instructed Dense query embeddings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import load_jsonl

from src.retrieval.catalog import Catalog
from src.retrieval.dense import DenseRetriever
from src.retrieval.embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    OpenAIEmbeddingEncoder,
    default_embedding_cache_dir,
)
from src.retrieval.evaluation.first_turn import (
    DEFAULT_KS,
    _official_initial_query,
    evaluate_recall,
)
from src.retrieval.query import build_retrieval_query
from src.retrieval.text import DEFAULT_TEXT_VERSION, TEXT_CONFIGS


QUERY_MODES = ("symmetric", "query", "query_instruction")


def compare_query_embedding_modes(
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    encoder: OpenAIEmbeddingEncoder,
    cache_dir: Path,
    *,
    text_version: str = DEFAULT_TEXT_VERSION,
    ks: Sequence[int] = DEFAULT_KS,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    query_batch_size: int | None = None,
) -> dict[str, Any]:
    queries = [
        build_retrieval_query(_official_initial_query(sample, catalog))
        for sample in samples
    ]
    modes: dict[str, Any] = {}
    for mode in QUERY_MODES:
        retriever = DenseRetriever(
            catalog,
            encoder,
            cache_dir,
            text_version=text_version,
            query_embedding_mode=mode,
            query_instruction=instruction if mode == "query_instruction" else None,
        )
        try:
            started = time.perf_counter()
            retriever.preload_queries(queries, batch_size=query_batch_size)
            preload_seconds = time.perf_counter() - started
            result = evaluate_recall(retriever, catalog, samples, ks=ks)
        finally:
            retriever.close()
        modes[mode] = {
            "text_type": "document" if mode == "symmetric" else "query",
            "instruction": instruction if mode == "query_instruction" else None,
            "query_preload_seconds": round(preload_seconds, 3),
            "query_preload_mean_ms": (
                round(preload_seconds * 1000.0 / len(queries), 3) if queries else 0.0
            ),
            **result,
        }
    return {
        "experiment": "dense_query_instruction_v1",
        "query_mode": "official_first_turn",
        "comparison_note": (
            "Query embeddings are preloaded; reported retrieval latency excludes API time."
        ),
        "modes": modes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Dense query text_type and instruction Recall@K."
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--text-version",
        choices=tuple(TEXT_CONFIGS),
        default=DEFAULT_TEXT_VERSION,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "dense_query_instruction_v1.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.query_batch_size is not None and args.query_batch_size <= 0:
        raise SystemExit("--query-batch-size must be positive")
    if not args.instruction.strip():
        raise SystemExit("--instruction must not be empty")

    started = time.perf_counter()
    catalog = Catalog.load(args.catalog)
    catalog_load_seconds = time.perf_counter() - started
    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    encoder = OpenAIEmbeddingEncoder.from_env()
    cache_dir = args.cache_dir or PROJECT_ROOT / default_embedding_cache_dir(
        encoder.model,
        encoder.dimension,
        args.text_version,
    )
    result = compare_query_embedding_modes(
        catalog,
        samples,
        encoder,
        cache_dir,
        text_version=args.text_version,
        ks=args.ks,
        instruction=args.instruction,
        query_batch_size=args.query_batch_size,
    )
    result = {
        **result,
        "text_version": args.text_version,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "catalog_load_seconds": round(catalog_load_seconds, 3),
        "embedding_cache": str(cache_dir),
        "embedding_model": encoder.model,
        "embedding_dimension": encoder.dimension,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
