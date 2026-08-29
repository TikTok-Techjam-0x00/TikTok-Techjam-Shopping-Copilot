"""Evaluate identity/needs Dense representations and their multi-vector fusion."""

from __future__ import annotations

import argparse
import json
import sys
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
from src.retrieval.evaluation.first_turn import DEFAULT_KS, _official_initial_query, evaluate_recall
from src.retrieval.experiments.dense_text import precompute_query_vectors
from src.retrieval.multivector import MultiVectorConfig, MultiVectorDenseRetriever
from src.retrieval.query import build_retrieval_query


IDENTITY_VERSION = "dense_identity_v1"
NEEDS_VERSION = "dense_needs_v1"


def compare_multivector_dense(
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    encoder: OpenAIEmbeddingEncoder,
    *,
    identity_weights: Sequence[float] = (0.3, 0.5, 0.7),
    ks: Sequence[int] = DEFAULT_KS,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    query_batch_size: int | None = None,
    cache_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Compare two single vectors, max fusion, and configured weighted fusions."""
    queries = [
        build_retrieval_query(_official_initial_query(sample, catalog))
        for sample in samples
    ]
    shared_encoder, preload_seconds = precompute_query_vectors(
        encoder,
        queries,
        instruction,
        batch_size=query_batch_size,
    )
    identity_cache = cache_root / default_embedding_cache_dir(
        encoder.model,
        encoder.dimension,
        IDENTITY_VERSION,
    )
    needs_cache = cache_root / default_embedding_cache_dir(
        encoder.model,
        encoder.dimension,
        NEEDS_VERSION,
    )
    methods: dict[str, Any] = {}

    for name, version, cache_dir in (
        ("identity_only", IDENTITY_VERSION, identity_cache),
        ("needs_only", NEEDS_VERSION, needs_cache),
    ):
        retriever = DenseRetriever(
            catalog,
            shared_encoder,
            cache_dir,
            text_version=version,
            query_embedding_mode="query_instruction",
            query_instruction=instruction,
        )
        try:
            methods[name] = evaluate_recall(retriever, catalog, samples, ks=ks)
        finally:
            retriever.close()

    fusion_configs = [("max", MultiVectorConfig(fusion="max"))]
    fusion_configs.extend(
        (
            f"weighted_identity_{weight:.1f}",
            MultiVectorConfig(fusion="weighted", identity_weight=float(weight)),
        )
        for weight in identity_weights
    )
    for name, config in fusion_configs:
        retriever = MultiVectorDenseRetriever(
            catalog,
            shared_encoder,
            identity_cache,
            needs_cache,
            query_instruction=instruction,
            config=config,
        )
        try:
            methods[name] = evaluate_recall(retriever, catalog, samples, ks=ks)
        finally:
            retriever.close()

    return {
        "experiment": "dense_identity_needs_multivector_v1",
        "evaluation_mode": "official_first_turn_strict",
        "query_embedding_mode": "query_instruction",
        "query_instruction": instruction,
        "shared_unique_query_count": len(set(queries)),
        "shared_query_preload_seconds": round(preload_seconds, 3),
        "identity_cache": str(identity_cache),
        "needs_cache": str(needs_cache),
        "identity_weights": [float(value) for value in identity_weights],
        "methods": methods,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate identity/needs multi-vector Dense Retrieval.")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--identity-weights", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "retrieval" / "dense_multivector_v1.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.query_batch_size is not None and args.query_batch_size <= 0:
        raise SystemExit("--query-batch-size must be positive")
    if any(not 0.0 <= weight <= 1.0 for weight in args.identity_weights):
        raise SystemExit("--identity-weights must stay between 0 and 1")
    if any(k <= 0 for k in args.ks):
        raise SystemExit("--ks values must be positive")
    if not args.instruction.strip():
        raise SystemExit("--instruction must not be empty")

    catalog = Catalog.load(args.catalog)
    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    encoder = OpenAIEmbeddingEncoder.from_env()
    result = compare_multivector_dense(
        catalog,
        samples,
        encoder,
        identity_weights=args.identity_weights,
        ks=args.ks,
        instruction=args.instruction,
        query_batch_size=args.query_batch_size,
    )
    result = {
        **result,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "sample_count": len(samples),
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
