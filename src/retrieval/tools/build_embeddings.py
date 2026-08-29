"""Build the 50,000-product Dense Retrieval embedding cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import Catalog
from src.retrieval.embedding import (
    OpenAIEmbeddingEncoder,
    build_embedding_cache,
    default_embedding_cache_dir,
)
from src.retrieval.text import DEFAULT_TEXT_VERSION, TEXT_CONFIGS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch encode the frozen catalog and cache normalized embeddings."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "catalog.jsonl",
    )
    parser.add_argument(
        "--text-version",
        choices=tuple(TEXT_CONFIGS),
        default=DEFAULT_TEXT_VERSION,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API requests; progress is committed in catalog order.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard incompatible/complete cache contents and rebuild.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    started = time.perf_counter()
    catalog = Catalog.load(args.catalog)
    catalog_load_seconds = time.perf_counter() - started
    encoder = OpenAIEmbeddingEncoder.from_env()
    cache_dir = args.cache_dir or PROJECT_ROOT / default_embedding_cache_dir(
        encoder.model,
        encoder.dimension,
        args.text_version,
    )

    last_reported = -1

    def report(completed: int, total: int) -> None:
        nonlocal last_reported
        percentage = int(completed * 100 / total) if total else 100
        if completed in (0, total) or percentage >= last_reported + 1:
            last_reported = percentage
            print(f"embedding progress: {completed}/{total} ({percentage}%)", flush=True)

    build_started = time.perf_counter()
    cache = build_embedding_cache(
        catalog,
        encoder,
        cache_dir,
        text_version=args.text_version,
        batch_size=args.batch_size,
        workers=args.workers,
        force=args.force,
        progress=report,
    )
    result = {
        "catalog": str(args.catalog),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "catalog_load_seconds": round(catalog_load_seconds, 3),
        "cache_dir": str(cache.cache_dir),
        "cache_hit": cache.cache_hit,
        "embedding_build_seconds": round(time.perf_counter() - build_started, 3),
        "manifest": asdict(cache.manifest),
        "matrix_shape": list(cache.embeddings.shape),
        "matrix_dtype": str(cache.embeddings.dtype),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
