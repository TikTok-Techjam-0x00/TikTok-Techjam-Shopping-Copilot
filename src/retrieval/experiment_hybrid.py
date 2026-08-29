"""Compare BM25, Dense, candidate union, RRF, and weighted fusion recall."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import load_jsonl

from src.item import Candidate
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.catalog import Catalog
from src.retrieval.dense import QUERY_EMBEDDING_MODES, DenseRetriever
from src.retrieval.embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    OpenAIEmbeddingEncoder,
    default_embedding_cache_dir,
)
from src.retrieval.evaluate import (
    DEFAULT_KS,
    _official_initial_query,
    _percentile,
    _recall_summary,
    _target_asin,
    _validated_ks,
)
from src.retrieval.hybrid import (
    candidate_union,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from src.retrieval.query import build_retrieval_query
from src.retrieval.retriever import RetrievalStrategy
from src.retrieval.text import DEFAULT_TEXT_VERSION, TEXT_CONFIGS


METHODS = ("bm25", "dense", "union", "rrf", "weighted")


def _target_rank(candidates: Sequence[Candidate], target: str) -> int | None:
    seen: set[str] = set()
    for rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Candidate):
            raise TypeError("retrievers must return shared Candidate objects")
        if candidate.parent_asin in seen:
            raise ValueError("retriever returned duplicate parent_asin values")
        seen.add(candidate.parent_asin)
        if candidate.parent_asin == target:
            return rank
    return None


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    latency_ms = [value * 1000.0 for value in values]
    return {
        "mean": round(statistics.fmean(latency_ms), 3) if latency_ms else 0.0,
        "p50": round(_percentile(latency_ms, 0.50), 3),
        "p95": round(_percentile(latency_ms, 0.95), 3),
        "max": round(max(latency_ms), 3) if latency_ms else 0.0,
    }


def compare_hybrid_methods(
    bm25: RetrievalStrategy,
    dense: RetrievalStrategy,
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    rank_constant: float = 60.0,
    alpha: float = 0.5,
    preload_dense_queries: bool = True,
    query_batch_size: int | None = None,
) -> dict[str, Any]:
    """Run both source retrievers once per query and derive all five methods.

    ``union Recall@K`` means the target occurs in BM25 Top-K or Dense Top-K.
    Its physical pool can contain up to 2K unique products. RRF and weighted
    results are strict ranked Top-K lists and are directly comparable to the
    source Top-K results.
    """
    recall_ks = _validated_ks(ks)
    max_k = recall_ks[-1]
    prepared: list[tuple[Mapping[str, Any], str, str, str]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample {index} must be an object")
        target = _target_asin(sample)
        raw_query = _official_initial_query(sample, catalog)
        query = build_retrieval_query(raw_query)
        scenario = str(sample.get("scenario_type") or "unknown")
        prepared.append((sample, target, query, scenario))

    preload_seconds = 0.0
    preload = getattr(dense, "preload_queries", None)
    if preload_dense_queries and callable(preload):
        started = time.perf_counter()
        preload(
            [query for _, _, query, _ in prepared],
            batch_size=query_batch_size,
        )
        preload_seconds = time.perf_counter() - started

    ranks: dict[str, list[int | None]] = {method: [] for method in METHODS}
    scenario_ranks: dict[str, dict[str, list[int | None]]] = {
        method: defaultdict(list) for method in METHODS
    }
    latencies: dict[str, list[float]] = {method: [] for method in METHODS}
    union_sizes: dict[int, list[int]] = {k: [] for k in recall_ks}

    for _, target, query, scenario in prepared:
        started = time.perf_counter()
        bm25_candidates = bm25.retrieve(query, k=max_k)
        latencies["bm25"].append(time.perf_counter() - started)

        started = time.perf_counter()
        dense_candidates = dense.retrieve(query, k=max_k)
        latencies["dense"].append(time.perf_counter() - started)

        bm25_rank = _target_rank(bm25_candidates, target)
        dense_rank = _target_rank(dense_candidates, target)
        source_ranks = [rank for rank in (bm25_rank, dense_rank) if rank is not None]
        union_rank = min(source_ranks) if source_ranks else None

        started = time.perf_counter()
        full_union = candidate_union(bm25_candidates, dense_candidates)
        latencies["union"].append(time.perf_counter() - started)

        started = time.perf_counter()
        rrf_candidates = reciprocal_rank_fusion(
            bm25_candidates,
            dense_candidates,
            k=max_k,
            rank_constant=rank_constant,
        )
        latencies["rrf"].append(time.perf_counter() - started)

        started = time.perf_counter()
        weighted_candidates = weighted_score_fusion(
            bm25_candidates,
            dense_candidates,
            k=max_k,
            alpha=alpha,
        )
        latencies["weighted"].append(time.perf_counter() - started)

        method_ranks = {
            "bm25": bm25_rank,
            "dense": dense_rank,
            "union": union_rank,
            "rrf": _target_rank(rrf_candidates, target),
            "weighted": _target_rank(weighted_candidates, target),
        }
        for method, rank in method_ranks.items():
            ranks[method].append(rank)
            scenario_ranks[method][scenario].append(rank)
        for k in recall_ks:
            union_sizes[k].append(
                len(candidate_union(bm25_candidates[:k], dense_candidates[:k]))
            )

        if len(full_union) > 2 * max_k:
            raise AssertionError("candidate union exceeded its 2K bound")

    return {
        "query_mode": "official_first_turn",
        "ks": list(recall_ks),
        "config": {
            "source_k": max_k,
            "rank_constant": rank_constant,
            "weighted_alpha": alpha,
            "union_semantics": "target in BM25 Top-K OR Dense Top-K; pool size <= 2K",
            "latency_semantics": (
                "dense query embeddings are batch-preloaded; dense query latency "
                "reports matrix search only"
            ),
        },
        "dense_query_preload_seconds": round(preload_seconds, 3),
        "dense_query_preload_mean_ms": (
            round(preload_seconds * 1000.0 / len(prepared), 3) if prepared else 0.0
        ),
        "methods": {
            method: {
                "overall": _recall_summary(ranks[method], recall_ks),
                "scenario_metrics": {
                    scenario: _recall_summary(values, recall_ks)
                    for scenario, values in sorted(scenario_ranks[method].items())
                },
                "query_latency_ms": _latency_summary(latencies[method]),
            }
            for method in METHODS
        },
        "union_pool_size": {
            f"source_top_{k}": {
                "mean_unique": round(statistics.fmean(values), 3) if values else 0.0,
                "max_unique": max(values) if values else 0,
            }
            for k, values in union_sizes.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare BM25, Dense, Union, RRF, and weighted fusion Recall@K."
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
        default=None,
        help="Legacy shortcut: use one text version for both BM25 and Dense.",
    )
    parser.add_argument(
        "--bm25-text-version",
        choices=tuple(TEXT_CONFIGS),
        default=DEFAULT_TEXT_VERSION,
    )
    parser.add_argument(
        "--dense-text-version",
        choices=tuple(TEXT_CONFIGS),
        default=DEFAULT_TEXT_VERSION,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument(
        "--query-embedding-mode",
        choices=tuple(sorted(QUERY_EMBEDDING_MODES)),
        default="query_instruction",
    )
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "retrieval_hybrid_comparison.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.query_batch_size is not None and args.query_batch_size <= 0:
        raise SystemExit("--query-batch-size must be positive")

    started = time.perf_counter()
    catalog = Catalog.load(args.catalog)
    catalog_load_seconds = time.perf_counter() - started
    encoder = OpenAIEmbeddingEncoder.from_env()
    bm25_text_version = args.text_version or args.bm25_text_version
    dense_text_version = args.text_version or args.dense_text_version
    cache_dir = args.cache_dir or PROJECT_ROOT / default_embedding_cache_dir(
        encoder.model,
        encoder.dimension,
        dense_text_version,
    )
    bm25 = BM25Retriever(catalog, text_version=bm25_text_version)
    try:
        dense = DenseRetriever(
            catalog,
            encoder,
            cache_dir,
            text_version=dense_text_version,
            query_embedding_mode=args.query_embedding_mode,
            query_instruction=(
                args.query_instruction
                if args.query_embedding_mode == "query_instruction"
                else None
            ),
        )
        samples = load_jsonl(args.dataset)
        if args.limit is not None:
            samples = samples[: args.limit]
        try:
            result = compare_hybrid_methods(
                bm25,
                dense,
                catalog,
                samples,
                ks=args.ks,
                rank_constant=args.rrf_constant,
                alpha=args.alpha,
                query_batch_size=args.query_batch_size,
            )
        finally:
            dense.close()
    finally:
        bm25.close()

    result = {
        "experiment": (
            "hybrid_qwen_query_instruction_v2"
            if args.query_embedding_mode == "query_instruction"
            else "hybrid_qwen_v1"
        ),
        "bm25_text_version": bm25_text_version,
        "dense_text_version": dense_text_version,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "catalog_load_seconds": round(catalog_load_seconds, 3),
        "embedding_cache": str(cache_dir),
        "embedding_model": encoder.model,
        "embedding_dimension": encoder.dimension,
        "query_embedding_mode": args.query_embedding_mode,
        "query_instruction": (
            args.query_instruction
            if args.query_embedding_mode == "query_instruction"
            else None
        ),
        **result,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
