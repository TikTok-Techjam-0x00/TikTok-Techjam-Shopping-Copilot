"""Export canonical derived product attributes from the frozen catalog.

Run directly from VSCode or PowerShell:

    python extract_attributes.py

The default output contains ``parent_asin`` and derived ``attributes`` only.
Generated output is ignored by Git because it can be rebuilt from the catalog.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from src.attribute import attribute_map_to_dict
from src.item import Item


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "catalog.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "catalog_attributes.jsonl"


@contextmanager
def _text_reader(path: Path) -> Iterator[TextIO]:
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield handle


@contextmanager
def _text_writer(path: Path) -> Iterator[TextIO]:
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            yield handle


def export_catalog_attributes(
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Stream catalog rows to a reproducible ASIN-to-attributes JSONL file."""
    source = Path(catalog_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("output path must be different from the catalog path")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        destination.with_name(destination.stem + ".tmp" + destination.suffix)
        if destination.suffix.casefold() == ".gz"
        else destination.with_name(destination.name + ".tmp")
    )
    rows_seen = 0
    items_written = 0
    malformed_rows = 0
    duplicate_asins = 0
    seen: set[str] = set()

    try:
        with _text_reader(source) as source_file, _text_writer(temporary) as output_file:
            for line in source_file:
                if limit is not None and items_written >= limit:
                    break
                if not line.strip():
                    continue
                rows_seen += 1
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("catalog row must be an object")
                    product = Item.from_dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    malformed_rows += 1
                    continue
                if product.parent_asin in seen:
                    duplicate_asins += 1
                    continue
                seen.add(product.parent_asin)
                output_file.write(
                    json.dumps(
                        {
                            "parent_asin": product.parent_asin,
                            "attributes": attribute_map_to_dict(product.attributes),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                items_written += 1
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "rows_seen": rows_seen,
        "items_written": items_written,
        "malformed_rows": malformed_rows,
        "duplicate_asins": duplicate_asins,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 catalog 的统一商品 attributes")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="只导出前 N 个有效商品")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stats = export_catalog_attributes(args.catalog, args.output, limit=args.limit)
    print(
        json.dumps(
            {
                "catalog": str(args.catalog.resolve()),
                "output": str(args.output.resolve()),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
