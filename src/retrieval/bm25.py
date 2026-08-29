"""Reusable SQLite FTS5 BM25 baseline for Module 1."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from ..item import Candidate, Candidates100, Item
from .catalog import Catalog
from .query import build_retrieval_query
from .text import (
    DEFAULT_TEXT_VERSION,
    ProductTextConfig,
    SEARCH_FIELDS,
    build_bm25_fields,
    resolve_text_config,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "i", "in", "is", "it", "me", "my", "of", "on", "or",
        "please", "some", "that", "the", "this", "to", "want", "with",
        "would", "you", "looking",
    }
)


def _terms(text: str, limit: int) -> list[str]:
    terms = (
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.casefold() not in STOPWORDS
    )
    return list(dict.fromkeys(terms))[:limit]


@dataclass(frozen=True, slots=True)
class BM25Weights:
    """FTS5 column weights, excluding the unindexed ``parent_asin`` column."""

    title: float = 6.0
    categories: float = 4.0
    features: float = 2.5
    attributes: float = 2.5
    details: float = 2.5
    store: float = 1.5
    description: float = 1.0

    def sql_parameters(self, active_fields: tuple[str, ...]) -> tuple[float, ...]:
        active = set(active_fields)
        return (
            0.0,
            *(getattr(self, field) if field in active else 0.0 for field in SEARCH_FIELDS),
        )


class BM25Retriever:
    """In-memory lexical retriever returning full shared ``Candidate`` objects.

    Native SQLite BM25 is lower-is-better. This component negates it so both
    ``bm25_score`` and ``retrieval_score`` are higher-is-better, matching 3A's
    documented input contract.
    """

    def __init__(
        self,
        catalog: Catalog | str,
        *,
        weights: BM25Weights | None = None,
        max_query_terms: int = 40,
        text_version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
    ) -> None:
        self.catalog = catalog if isinstance(catalog, Catalog) else Catalog.load(catalog)
        self.weights = weights or BM25Weights()
        self.max_query_terms = max(1, int(max_query_terms))
        self.text_config = resolve_text_config(text_version)
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, "
            + ", ".join(SEARCH_FIELDS)
            + ", "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        def rows():
            for product in self.catalog.items_in_order:
                fields = build_bm25_fields(product, self.text_config)
                yield (
                    product.parent_asin,
                    *(fields[field] for field in SEARCH_FIELDS),
                )

        placeholders = ", ".join("?" for _ in range(1 + len(SEARCH_FIELDS)))
        cursor.executemany(f"INSERT INTO products VALUES ({placeholders})", rows())
        self.connection.commit()

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        """Retrieve unique candidates ordered best-to-worst."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            return []

        retrieval_query = build_retrieval_query(query, state, intent)
        terms = _terms(retrieval_query, self.max_query_terms)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        weight_parameters = self.weights.sql_parameters(self.text_config.fields)
        placeholders = ", ".join("?" for _ in weight_parameters)
        sql = (
            "SELECT parent_asin, bm25(products, "
            + placeholders
            + ") AS native_score FROM products WHERE products MATCH ? "
            "ORDER BY native_score ASC, rowid ASC LIMIT ?"
        )
        parameters = (*weight_parameters, expression, k)
        rows = self.connection.execute(sql, parameters).fetchall()

        candidates: Candidates100 = []
        for rank, (parent_asin, native_score) in enumerate(rows, start=1):
            product: Item | None = self.catalog.get(str(parent_asin))
            if product is None:
                continue
            score = -float(native_score)
            candidates.append(
                Candidate(
                    item=product,
                    bm25_score=score,
                    retrieval_score=score,
                    retrieval_rank=rank,
                )
            )
        return candidates

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> BM25Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
