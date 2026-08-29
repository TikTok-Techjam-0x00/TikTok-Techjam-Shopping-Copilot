"""Compare Dense product-text caches with one shared instructed query matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


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
from src.retrieval.query import build_retrieval_query
from src.retrieval.text import TEXT_CONFIGS


DEFAULT_VERSIONS = (
    "all_fields_v4",
    "dense_attributes_v2",
    "dense_attributes_v2_unlabeled",
)


class PrecomputedQueryEncoder:
    """Serve one shared query-vector set to several document-text caches."""

    def __init__(
        self,
        model: str,
        dimension: int,
        vectors: Mapping[str, np.ndarray],
        instruction: str,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.batch_size = max(1, len(vectors))
        self._vectors = vectors
        self._instruction = instruction

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise RuntimeError("document encoding is not available in an evaluation cache")

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        instruct: str | None = None,
    ) -> np.ndarray:
        if instruct != self._instruction:
            raise ValueError("query instruction differs from the precomputed experiment")
        try:
            return np.asarray([self._vectors[text] for text in texts], dtype=np.float32)
        except KeyError as error:
            raise ValueError(f"query was not precomputed: {error.args[0]!r}") from error


def precompute_query_vectors(
    encoder: OpenAIEmbeddingEncoder,
    queries: Sequence[str],
    instruction: str,
    *,
    batch_size: int | None = None,
) -> tuple[PrecomputedQueryEncoder, float]:
    unique = list(dict.fromkeys(query for query in queries if query))
    size = int(batch_size or encoder.batch_size)
    if size <= 0:
        raise ValueError("batch_size must be positive")
    vectors: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for offset in range(0, len(unique), size):
        batch = unique[offset : offset + size]
        encoded = encoder.encode_queries(batch, instruct=instruction)
        if encoded.shape != (len(batch), encoder.dimension):
            raise ValueError("query embedding provider returned an unexpected shape")
        vectors.update(zip(batch, encoded, strict=True))
    elapsed = time.perf_counter() - started
    return (
        PrecomputedQueryEncoder(
            encoder.model,
            encoder.dimension,
            vectors,
            instruction,
        ),
        elapsed,
    )


def compare_dense_text_versions(
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    encoder: OpenAIEmbeddingEncoder,
    *,
    versions: Sequence[str] = DEFAULT_VERSIONS,
    ks: Sequence[int] = DEFAULT_KS,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
    query_batch_size: int | None = None,
    cache_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
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
    results: dict[str, Any] = {}
    for version in versions:
        cache_dir = cache_root / default_embedding_cache_dir(
            encoder.model,
            encoder.dimension,
            version,
        )
        retriever = DenseRetriever(
            catalog,
            shared_encoder,
            cache_dir,
            text_version=version,
            query_embedding_mode="query_instruction",
            query_instruction=instruction,
        )
        try:
            retriever.preload_queries(queries, batch_size=len(queries) or 1)
            results[version] = {
                "embedding_cache": str(cache_dir),
                **evaluate_recall(retriever, catalog, samples, ks=ks),
            }
        finally:
            retriever.close()
    return {
        "experiment": "dense_product_text_query_instruction_v1",
        "query_embedding_mode": "query_instruction",
        "query_instruction": instruction,
        "shared_unique_query_count": len(set(queries)),
        "shared_query_preload_seconds": round(preload_seconds, 3),
        "versions": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare strict Dense Recall for product-text versions."
    )
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--versions", nargs="+", choices=tuple(TEXT_CONFIGS), default=list(DEFAULT_VERSIONS))
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "dense_product_text_query_instruction_v1.json",
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
    catalog = Catalog.load(args.catalog)
    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    encoder = OpenAIEmbeddingEncoder.from_env()
    result = compare_dense_text_versions(
        catalog,
        samples,
        encoder,
        versions=args.versions,
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
