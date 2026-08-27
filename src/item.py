"""Shared product objects exchanged between retrieval and reranking.

The lowercase class names are intentional: they match the team-agreed pipeline
contract (`item` -> `candidate` -> `reranked_candidate`).
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


@dataclass(slots=True)
class item(Mapping[str, Any]):
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
    store: str = ""

    def __post_init__(self) -> None:
        self.parent_asin = str(self.parent_asin).strip()
        if not self.parent_asin:
            raise ValueError("parent_asin must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> item:
        """Build an item from one JSONL catalog record."""
        details = value.get("details")
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
            store=str(value.get("store") or ""),
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

    # Mapping support lets existing dictionary-based modules read these objects
    # during the migration without duplicating product data under `product`.
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(slots=True)
class candidate(item):
    """Retrieval output item; a `candidates_100` entry."""

    retrieval_score: float | None = None
    retrieval_rank: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> candidate:
        nested = value.get("product")
        product = dict(nested) if isinstance(nested, Mapping) else dict(value)
        if not product.get("parent_asin"):
            product["parent_asin"] = value.get("parent_asin")
        base = item.from_dict(product)
        return cls(
            **base.to_dict(),
            retrieval_score=_optional_float(value.get("retrieval_score")),
            retrieval_rank=_optional_int(value.get("retrieval_rank")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = super(candidate, self).to_dict()
        result.update(
            {
                "retrieval_score": self.retrieval_score,
                "retrieval_rank": self.retrieval_rank,
            }
        )
        return result


@dataclass(slots=True)
class reranked_candidate(item):
    """Reranking output item; a `candidates_10` entry."""

    retrieval_score: float | None = None
    retrieval_rank: int | None = None
    rank: int = 0
    score: float = 0.0
    matched: list[str] = field(default_factory=list)
    violation: list[str] = field(default_factory=list)

    @classmethod
    def from_candidate(
        cls,
        source: candidate,
        *,
        rank: int,
        score: float,
        matched: Sequence[str],
        violation: Sequence[str],
    ) -> reranked_candidate:
        return cls(
            **item.to_dict(source),
            retrieval_score=source.retrieval_score,
            retrieval_rank=source.retrieval_rank,
            rank=rank,
            score=score,
            matched=list(matched),
            violation=list(violation),
        )

    @property
    def matched_attributes(self) -> list[str]:
        """Compatibility name used by the first ranking prototype."""
        return self.matched

    @property
    def violations(self) -> list[str]:
        """Compatibility name used by the first ranking prototype."""
        return self.violation

    def to_dict(self) -> dict[str, Any]:
        result = super(reranked_candidate, self).to_dict()
        result.update(
            {
                "retrieval_score": self.retrieval_score,
                "retrieval_rank": self.retrieval_rank,
                "rank": self.rank,
                "score": self.score,
                "matched": list(self.matched),
                "violation": list(self.violation),
            }
        )
        return result


candidates_100: TypeAlias = list[candidate]
candidates_10: TypeAlias = list[reranked_candidate]

# Optional conventional aliases for teammates who prefer PEP 8 class names.
Item = item
Candidate = candidate
RerankedCandidate = reranked_candidate


__all__ = [
    "DATASET_FIELDS",
    "item",
    "candidate",
    "reranked_candidate",
    "candidates_100",
    "candidates_10",
    "Item",
    "Candidate",
    "RerankedCandidate",
]
