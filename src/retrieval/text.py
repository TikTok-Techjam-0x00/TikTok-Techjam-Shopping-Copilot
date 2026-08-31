"""Versioned product text construction for lexical and dense retrieval.

BM25 consumes the returned fields separately so column weights remain useful.
Dense retrieval can consume the labeled string returned by
``build_product_text``.  Keeping both paths on one version registry makes an
embedding cache reproducible: model + catalog + text version identify its
contents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..attribute import (
    AttributeName,
    product_attribute_text,
    product_attribute_values,
)
from ..item import Item


SEARCH_FIELDS = (
    "title",
    "categories",
    "features",
    "attributes",
    "details",
    "store",
    "description",
)

_ATTRIBUTE_TEXT_ORDER = (
    AttributeName.MATERIAL,
    AttributeName.COLOR,
    AttributeName.SIZE,
    AttributeName.STYLE,
    AttributeName.BRAND,
    AttributeName.USE_CASE,
)

_FIELD_LABELS = {
    "title": "Title",
    "categories": "Category",
    "features": "Features",
    "attributes": "Attributes",
    "details": "Details",
    "store": "Store",
    "description": "Description",
}


@dataclass(frozen=True, slots=True)
class ProductTextConfig:
    """One reproducible set of catalog fields used as retrieval text."""

    name: str
    fields: tuple[str, ...]
    description: str
    layout: str = "generic_labeled"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("text config name must not be empty")
        if not self.fields:
            raise ValueError("text config must include at least one field")
        unknown = set(self.fields) - set(SEARCH_FIELDS)
        if unknown:
            raise ValueError(f"unknown product text fields: {sorted(unknown)}")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("product text fields must be unique")
        if self.layout not in {"generic_labeled", "dense_needs"}:
            raise ValueError(f"unknown product text layout: {self.layout!r}")


TEXT_CONFIGS: dict[str, ProductTextConfig] = {
    "title_category_v1": ProductTextConfig(
        name="title_category_v1",
        fields=("title", "categories"),
        description="Title and catalog category hierarchy.",
    ),
    "all_fields_v4": ProductTextConfig(
        name="all_fields_v4",
        fields=("title", "categories", "features", "details", "store", "description"),
        description="Current BM25 baseline: all searchable official text fields.",
    ),
    "dense_needs_v1": ProductTextConfig(
        name="dense_needs_v1",
        fields=("features", "attributes"),
        description="Needs vector: material, color, size, style, use case, and features.",
        layout="dense_needs",
    ),
}

DEFAULT_TEXT_VERSION = "all_fields_v4"


def resolve_text_config(
    version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
) -> ProductTextConfig:
    """Resolve a registered name while still allowing explicit test configs."""
    if isinstance(version, ProductTextConfig):
        return version
    try:
        return TEXT_CONFIGS[str(version)]
    except KeyError as error:
        choices = ", ".join(TEXT_CONFIGS)
        raise ValueError(f"unknown text version {version!r}; choose one of: {choices}") from error


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            part
            for key, entry in value.items()
            if (part := f"{key} {_flatten(entry)}".strip())
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(part for entry in value if (part := _flatten(entry)))
    return " ".join(str(value).split())


def _selected_attribute_text(item: Item) -> str:
    parts: list[str] = []
    for name in _ATTRIBUTE_TEXT_ORDER:
        value = product_attribute_text(item.attributes, name)
        if value:
            parts.append(f"{name.value} {value}")
    return " ".join(parts)


def _attribute_text(
    item: Item,
    name: AttributeName,
    *,
    include_details: bool = True,
) -> str:
    values = product_attribute_values(
        item.attributes,
        name,
        include_details=include_details,
    )
    return _flatten(values)


def _limited(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(" ", 1)[0].strip()


def _dense_needs_parts(item: Item) -> list[tuple[str, str]]:
    return [
        ("Material", _attribute_text(item, AttributeName.MATERIAL)),
        ("Color", _attribute_text(item, AttributeName.COLOR)),
        # Exclude package/product measurement details from label-size semantics.
        ("Size", _attribute_text(item, AttributeName.SIZE, include_details=False)),
        ("Style", _attribute_text(item, AttributeName.STYLE)),
        ("Use case", _attribute_text(item, AttributeName.USE_CASE)),
        ("Features", _limited(_attribute_text(item, AttributeName.FEATURE), 1_200)),
    ]


def _render_parts(parts: Sequence[tuple[str, str]]) -> str:
    return "\n".join(
        f"{label}: {value}"
        for label, value in parts
        if value
    )


def product_field_text(item: Item, field: str) -> str:
    """Build one deterministic field without changing the official Item."""
    if field == "title":
        return _flatten(item.title)
    if field == "categories":
        return _flatten(item.categories)
    if field == "features":
        return _flatten(item.features)
    if field == "attributes":
        return _selected_attribute_text(item)
    if field == "details":
        return _flatten(item.details)
    if field == "store":
        return _flatten(item.store)
    if field == "description":
        return _flatten(item.description)
    raise ValueError(f"unknown product text field: {field!r}")


def build_bm25_fields(
    item: Item,
    version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
) -> dict[str, str]:
    """Return all stable FTS columns; excluded version fields are empty."""
    config = resolve_text_config(version)
    active = set(config.fields)
    return {
        field: product_field_text(item, field) if field in active else ""
        for field in SEARCH_FIELDS
    }


def build_product_text(
    item: Item,
    version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
) -> str:
    """Reproduce the exact labeled text used to fingerprint the vector cache."""
    config = resolve_text_config(version)
    if config.layout == "dense_needs":
        return _render_parts(_dense_needs_parts(item))
    lines = []
    for field in config.fields:
        value = product_field_text(item, field)
        if value:
            lines.append(f"{_FIELD_LABELS[field]}: {value}")
    return "\n".join(lines)


__all__ = [
    "SEARCH_FIELDS",
    "ProductTextConfig",
    "TEXT_CONFIGS",
    "DEFAULT_TEXT_VERSION",
    "resolve_text_config",
    "product_field_text",
    "build_bm25_fields",
    "build_product_text",
]
