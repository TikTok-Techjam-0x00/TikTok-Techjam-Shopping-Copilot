"""Shared product and pipeline objects.

`Item` is the catalog entity. Retrieval and reranking metadata use composition:
`Candidate.item` and `RankedCandidate.item` both reference an `Item`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias


DATASET_FIELDS = (
    "parent_asin",
    "title",
    "features",
    "description",
    "price",
    "categories",
    "details",
    "average_rating",
    "rating_number",
    "store",
)


def _string_list(value: object) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(entry) for entry in value if entry is not None]
    if value in (None, ""):
        return []
    return [str(value)]


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


class _MappingView(Mapping[str, Any]):
    """Let pipeline objects work with existing dictionary-based consumers."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(slots=True)
class Item(_MappingView):
    """One product using exactly the participant-visible catalog fields."""

    parent_asin: str
    title: str = ""
    features: list[str] = field(default_factory=list)
    description: list[str] = field(default_factory=list)
    price: float | None = None
    categories: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    average_rating: float | None = None
    rating_number: int | None = None
    store: str | None = None

    def __post_init__(self) -> None:
        self.parent_asin = str(self.parent_asin).strip()
        if not self.parent_asin:
            raise ValueError("parent_asin must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Item:
        """Build an Item from one JSONL catalog record."""
        details = value.get("details")
        store = value.get("store")
        return cls(
            parent_asin=str(value.get("parent_asin") or ""),
            title=str(value.get("title") or ""),
            features=_string_list(value.get("features")),
            description=_string_list(value.get("description")),
            price=_optional_float(value.get("price")),
            categories=_string_list(value.get("categories")),
            details=dict(details) if isinstance(details, Mapping) else {},
            average_rating=_optional_float(value.get("average_rating")),
            rating_number=_optional_int(value.get("rating_number")),
            store=str(store) if store not in (None, "") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the original dataset shape, safe for JSON serialization."""
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "features": list(self.features),
            "description": list(self.description),
            "price": self.price,
            "categories": list(self.categories),
            "details": dict(self.details),
            "average_rating": self.average_rating,
            "rating_number": self.rating_number,
            "store": self.store,
        }


@dataclass(slots=True)
class Candidate(_MappingView):
    """Retrieval output: a catalog Item plus retrieval-stage metadata."""

    item: Item
    bm25_score: float | None = None
    dense_score: float | None = None
    retrieval_score: float | None = None
    retrieval_rank: int | None = None

    @property
    def parent_asin(self) -> str:
        return self.item.parent_asin

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Candidate:
        """Accept the canonical nested shape and legacy inline/product shapes."""
        nested_item = value.get("item")
        if isinstance(nested_item, Item):
            product = nested_item
        elif isinstance(nested_item, Mapping):
            product = Item.from_dict(nested_item)
        else:
            nested_product = value.get("product")
            raw_product = (
                dict(nested_product)
                if isinstance(nested_product, Mapping)
                else dict(value)
            )
            if not raw_product.get("parent_asin"):
                raw_product["parent_asin"] = value.get("parent_asin")
            product = Item.from_dict(raw_product)
        return cls(
            item=product,
            bm25_score=_optional_float(value.get("bm25_score")),
            dense_score=_optional_float(value.get("dense_score")),
            retrieval_score=_optional_float(value.get("retrieval_score")),
            retrieval_rank=_optional_int(value.get("retrieval_rank")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
        }


@dataclass(slots=True)
class RankedCandidate(_MappingView):
    """Reranking output: an Item plus retrieval and reranking metadata."""

    item: Item
    bm25_score: float | None = None
    dense_score: float | None = None
    retrieval_score: float | None = None
    retrieval_rank: int | None = None
    rerank_score: float = 0.0
    rerank_rank: int = 0
    matched: list[str] = field(default_factory=list)
    violation: list[str] = field(default_factory=list)

    @property
    def parent_asin(self) -> str:
        return self.item.parent_asin

    @property
    def score(self) -> float:
        """Compatibility alias; new code should use rerank_score."""
        return self.rerank_score

    @property
    def rank(self) -> int:
        """Compatibility alias; new code should use rerank_rank."""
        return self.rerank_rank

    @property
    def matched_attributes(self) -> list[str]:
        return self.matched

    @property
    def violations(self) -> list[str]:
        return self.violation

    @classmethod
    def from_candidate(
        cls,
        source: Candidate,
        *,
        rerank_rank: int,
        rerank_score: float,
        matched: Sequence[str],
        violation: Sequence[str],
    ) -> RankedCandidate:
        return cls(
            item=source.item,
            bm25_score=source.bm25_score,
            dense_score=source.dense_score,
            retrieval_score=source.retrieval_score,
            retrieval_rank=source.retrieval_rank,
            rerank_rank=rerank_rank,
            rerank_score=rerank_score,
            matched=list(matched),
            violation=list(violation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "matched": list(self.matched),
            "violation": list(self.violation),
        }


Candidates100: TypeAlias = list[Candidate]
Candidates10: TypeAlias = list[RankedCandidate]

# Transitional aliases for code written before the composition refactor.
item = Item
candidate = Candidate
reranked_candidate = RankedCandidate
candidates_100: TypeAlias = Candidates100
candidates_10: TypeAlias = Candidates10


__all__ = [
    "DATASET_FIELDS",
    "Item",
    "Candidate",
    "RankedCandidate",
    "Candidates100",
    "Candidates10",
    "item",
    "candidate",
    "reranked_candidate",
    "candidates_100",
    "candidates_10",
]
