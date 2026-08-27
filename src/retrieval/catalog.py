"""Load the frozen product catalog once and provide ASIN lookup."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..item import Item


@dataclass(frozen=True, slots=True)
class CatalogLoadStats:
    """Diagnostics for catalog normalization without failing the full pipeline."""

    rows_seen: int = 0
    items_loaded: int = 0
    duplicate_asins: int = 0
    malformed_rows: int = 0


class Catalog(Mapping[str, Item]):
    """An immutable-by-convention ``parent_asin -> Item`` catalog.

    Duplicate identifiers keep the first valid product so identity and retrieval
    output stay deterministic. Malformed JSON and products without an ASIN are
    skipped and counted in ``stats``.
    """

    def __init__(
        self,
        products: Mapping[str, Item],
        *,
        stats: CatalogLoadStats | None = None,
        source: Path | None = None,
    ) -> None:
        self._products = dict(products)
        self.stats = stats or CatalogLoadStats(items_loaded=len(self._products))
        self.source = source

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        """Parse a JSONL or JSONL.GZ catalog exactly once."""
        source = Path(path)
        products: dict[str, Item] = {}
        rows_seen = 0
        duplicates = 0
        malformed = 0

        opener = gzip.open if source.suffix.casefold() == ".gz" else Path.open
        open_args = {"mode": "rt", "encoding": "utf-8"} if opener is gzip.open else {"encoding": "utf-8"}
        with opener(source, **open_args) as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_seen += 1
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, Mapping):
                        raise TypeError("catalog row must be an object")
                    product = Item.from_dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    malformed += 1
                    continue
                if product.parent_asin in products:
                    duplicates += 1
                    continue
                products[product.parent_asin] = product

        return cls(
            products,
            stats=CatalogLoadStats(
                rows_seen=rows_seen,
                items_loaded=len(products),
                duplicate_asins=duplicates,
                malformed_rows=malformed,
            ),
            source=source,
        )

    @classmethod
    def from_items(cls, values: Iterable[Item | Mapping[str, object]]) -> Catalog:
        """Build a small in-memory catalog, mainly for tests and experiments."""
        products: dict[str, Item] = {}
        rows_seen = 0
        duplicates = 0
        malformed = 0
        for value in values:
            rows_seen += 1
            try:
                product = value if isinstance(value, Item) else Item.from_dict(value)
            except (TypeError, ValueError):
                malformed += 1
                continue
            if product.parent_asin in products:
                duplicates += 1
                continue
            products[product.parent_asin] = product
        return cls(
            products,
            stats=CatalogLoadStats(
                rows_seen=rows_seen,
                items_loaded=len(products),
                duplicate_asins=duplicates,
                malformed_rows=malformed,
            ),
        )

    def __getitem__(self, parent_asin: str) -> Item:
        return self._products[parent_asin]

    def __iter__(self) -> Iterator[str]:
        return iter(self._products)

    def __len__(self) -> int:
        return len(self._products)

    @property
    def items_in_order(self) -> tuple[Item, ...]:
        """Products in deterministic source order for index construction."""
        return tuple(self._products.values())
