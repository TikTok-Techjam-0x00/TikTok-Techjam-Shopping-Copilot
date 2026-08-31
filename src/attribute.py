"""Official ten-field attribute contract shared by the shopping pipeline.

``AttributeName`` deliberately matches the evaluator's ``ask_attribute``
vocabulary. Catalog/detail aliases are normalized into these ten fields, and
unrecognized source fields are retained under ``AttributeName.OTHER``.

The retired richer schema remains in the development history. Production
modules should import this file.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


class AttributeName(str, Enum):
    """The evaluator-compatible attribute vocabulary."""

    CATEGORY = "category"
    MATERIAL = "material"
    COLOR = "color"
    SIZE = "size"
    STYLE = "style"
    BRAND = "brand"
    BUDGET = "budget"
    FEATURE = "feature"
    USE_CASE = "use_case"
    OTHER = "other"

    def __str__(self) -> str:
        return self.value


ATTRIBUTE_NAMES: tuple[AttributeName, ...] = tuple(AttributeName)
OFFICIAL_ASK_ATTRIBUTES = frozenset(name.value for name in AttributeName)


# Catalog fields and common details keys are aliases, not additional public
# attributes. Their values are folded into one of the official ten fields.
_ATTRIBUTE_ALIASES: dict[str, AttributeName] = {
    # Product kind.
    "category": AttributeName.CATEGORY,
    "categories": AttributeName.CATEGORY,
    "product_category": AttributeName.CATEGORY,
    "product_type": AttributeName.CATEGORY,
    "item_type": AttributeName.CATEGORY,
    # Material.
    "material": AttributeName.MATERIAL,
    "materials": AttributeName.MATERIAL,
    "fabric": AttributeName.MATERIAL,
    "fabric_type": AttributeName.MATERIAL,
    "material_type": AttributeName.MATERIAL,
    "outer_material": AttributeName.MATERIAL,
    "inner_material": AttributeName.MATERIAL,
    "metal_type": AttributeName.MATERIAL,
    # Color.
    "color": AttributeName.COLOR,
    "colors": AttributeName.COLOR,
    "colour": AttributeName.COLOR,
    "colours": AttributeName.COLOR,
    # Label size and physical measurements.
    "size": AttributeName.SIZE,
    "sizes": AttributeName.SIZE,
    "sizing": AttributeName.SIZE,
    "shoe_width": AttributeName.SIZE,
    "dimension": AttributeName.SIZE,
    "dimensions": AttributeName.SIZE,
    "product_dimensions": AttributeName.SIZE,
    "package_dimensions": AttributeName.SIZE,
    "diameter": AttributeName.SIZE,
    "length": AttributeName.SIZE,
    "width": AttributeName.SIZE,
    "height": AttributeName.SIZE,
    "weight": AttributeName.SIZE,
    "item_weight": AttributeName.SIZE,
    # Style, fit, pattern, and visual form.
    "style": AttributeName.STYLE,
    "fit": AttributeName.STYLE,
    "fit_type": AttributeName.STYLE,
    "pattern": AttributeName.STYLE,
    "pattern_type": AttributeName.STYLE,
    "theme": AttributeName.STYLE,
    "shape": AttributeName.STYLE,
    "finish_type": AttributeName.STYLE,
    "department": AttributeName.STYLE,
    "neck_style": AttributeName.STYLE,
    "sleeve_type": AttributeName.STYLE,
    "heel_type": AttributeName.STYLE,
    # Brand / maker.
    "brand": AttributeName.BRAND,
    "brand_name": AttributeName.BRAND,
    "store": AttributeName.BRAND,
    "manufacturer": AttributeName.BRAND,
    # Budget / price.
    "budget": AttributeName.BUDGET,
    "price": AttributeName.BUDGET,
    "price_range": AttributeName.BUDGET,
    "cost": AttributeName.BUDGET,
    # Product capabilities and evaluator-default constraint types.
    "feature": AttributeName.FEATURE,
    "features": AttributeName.FEATURE,
    "special_feature": AttributeName.FEATURE,
    "special_features": AttributeName.FEATURE,
    "closure_type": AttributeName.FEATURE,
    "included_components": AttributeName.FEATURE,
    "target_user": AttributeName.FEATURE,
    "target_audience": AttributeName.FEATURE,
    "audience": AttributeName.FEATURE,
    "gender": AttributeName.FEATURE,
    "age": AttributeName.FEATURE,
    "age_range": AttributeName.FEATURE,
    "age_range_description": AttributeName.FEATURE,
    "suggested_users": AttributeName.FEATURE,
    "rating": AttributeName.FEATURE,
    "average_rating": AttributeName.FEATURE,
    "rating_number": AttributeName.FEATURE,
    "review_count": AttributeName.FEATURE,
    "quantity": AttributeName.FEATURE,
    "number_of_items": AttributeName.FEATURE,
    "item_package_quantity": AttributeName.FEATURE,
    "pack_size": AttributeName.FEATURE,
    # Activity, occasion, or intended situation.
    "use_case": AttributeName.USE_CASE,
    "usecase": AttributeName.USE_CASE,
    "occasion": AttributeName.USE_CASE,
    "sport": AttributeName.USE_CASE,
    "sport_type": AttributeName.USE_CASE,
    "activity": AttributeName.USE_CASE,
    "purpose": AttributeName.USE_CASE,
    # Canonical fallback. Serialization always uses this singular official key.
    "other": AttributeName.OTHER,
}


# Conservative fallbacks for products whose useful values are not structured
# in ``details``. Explicit catalog metadata always wins over these matches.
_PRODUCT_TEXT_VALUE_PATTERNS: dict[AttributeName, re.Pattern[str]] = {
    AttributeName.MATERIAL: re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|"
        r"suede|denim|canvas|mesh|rubber|stainless steel|sterling silver)\b",
        re.IGNORECASE,
    ),
    AttributeName.COLOR: re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gr[ae]y|purple|yellow|"
        r"orange|beige|silver|gold)\b",
        re.IGNORECASE,
    ),
    AttributeName.SIZE: re.compile(
        r"\b(xxs|xs|small|medium|large|xl|xxl|wide|narrow|petite|plus size)\b",
        re.IGNORECASE,
    ),
    AttributeName.STYLE: re.compile(
        r"\b(casual|formal|classic|modern|vintage|sporty|slim fit|regular fit|"
        r"relaxed fit|loose fit|solid|striped|plaid|floral|polka dot)\b",
        re.IGNORECASE,
    ),
    AttributeName.USE_CASE: re.compile(
        r"\b(hiking|running|gym|winter|outdoor|work)\b",
        re.IGNORECASE,
    ),
}

_GENERIC_CATEGORIES = frozenset({"clothing, shoes & jewelry"})
_SIZE_DETAIL_ONLY_KEYS = frozenset(
    {
        "dimension",
        "dimensions",
        "product_dimensions",
        "package_dimensions",
        "diameter",
        "length",
        "width",
        "height",
        "weight",
        "item_weight",
    }
)
_DETAIL_PRESERVING_FIELDS = frozenset(
    {AttributeName.FEATURE, AttributeName.OTHER}
)

_SPACE_RE = re.compile(r"\s+")
_NAME_RE = re.compile(r"[^a-z0-9]+")


def _clean_name(value: object) -> str:
    return _NAME_RE.sub("_", str(value).strip().lower()).strip("_")


def _clean_text(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value)).strip()


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_text_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, Mapping):
        return [_clean_text(item) for item in value.values() if item not in (None, "")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_clean_text(item) for item in value if item not in (None, "")]
    return [_clean_text(value)]


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


@dataclass(slots=True)
class AttributeValue(Mapping[str, Any]):
    """One stable value shape for text, numeric ranges, and subfields."""

    values: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    details: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = _unique(_as_text_list(self.values))
        self.minimum = _as_float(self.minimum)
        self.maximum = _as_float(self.maximum)
        self.unit = _clean_text(self.unit) if self.unit not in (None, "") else None
        self.details = {
            _clean_name(key): _unique(_as_text_list(value))
            for key, value in self.details.items()
            if _clean_name(key) and _as_text_list(value)
        }
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            self.minimum, self.maximum = self.maximum, self.minimum

    @classmethod
    def from_raw(cls, value: object) -> AttributeValue:
        """Normalize common LLM and dataset outputs into ``AttributeValue``."""
        if isinstance(value, AttributeValue):
            return value.copy()
        if isinstance(value, Mapping):
            minimum = value.get("minimum", value.get("min", value.get("price_min")))
            maximum = value.get("maximum", value.get("max", value.get("price_max")))
            unit = value.get("unit", value.get("currency"))
            raw_values = value.get("values", value.get("value", []))
            reserved = {
                "minimum",
                "min",
                "price_min",
                "maximum",
                "max",
                "price_max",
                "unit",
                "currency",
                "values",
                "value",
                "details",
            }
            details: dict[str, object] = {}
            nested_details = value.get("details")
            if isinstance(nested_details, Mapping):
                details.update(nested_details)
            details.update(
                (str(key), entry)
                for key, entry in value.items()
                if str(key) not in reserved
            )
            return cls(
                values=_as_text_list(raw_values),
                minimum=_as_float(minimum),
                maximum=_as_float(maximum),
                unit=str(unit) if unit not in (None, "") else None,
                details={key: _as_text_list(entry) for key, entry in details.items()},
            )
        return cls(values=_as_text_list(value))

    @classmethod
    def range(
        cls,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        unit: str | None = None,
    ) -> AttributeValue:
        return cls(minimum=minimum, maximum=maximum, unit=unit)

    def copy(self) -> AttributeValue:
        return AttributeValue(
            values=list(self.values),
            minimum=self.minimum,
            maximum=self.maximum,
            unit=self.unit,
            details={key: list(values) for key, values in self.details.items()},
        )

    def merge(self, other: AttributeValue) -> None:
        """Merge another extraction without losing earlier information."""
        self.values = _unique([*self.values, *other.values])
        if other.minimum is not None:
            self.minimum = other.minimum
        if other.maximum is not None:
            self.maximum = other.maximum
        if other.unit is not None:
            self.unit = other.unit
        for key, values in other.details.items():
            self.details[key] = _unique([*self.details.get(key, []), *values])
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            self.minimum, self.maximum = self.maximum, self.minimum

    def is_empty(self) -> bool:
        return not (
            self.values
            or self.minimum is not None
            or self.maximum is not None
            or self.details
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "min": self.minimum,
            "max": self.maximum,
            "unit": self.unit,
            "details": {key: list(values) for key, values in self.details.items()},
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


AttributeMap: TypeAlias = dict[AttributeName, AttributeValue]


def normalize_attribute_name(value: object) -> AttributeName:
    """Map a catalog/LLM name to one official field; unknown names use OTHER."""
    if isinstance(value, AttributeName):
        return value
    return _ATTRIBUTE_ALIASES.get(_clean_name(value), AttributeName.OTHER)


def normalize_attribute_map(
    raw: Mapping[str | AttributeName, object] | None,
) -> AttributeMap:
    """Normalize constraints while retaining unknown fields under ``other``."""
    result: AttributeMap = {}
    if not isinstance(raw, Mapping):
        return result

    for raw_name, raw_value in raw.items():
        cleaned_name = _clean_name(raw_name)
        name = normalize_attribute_name(raw_name)
        if name is AttributeName.OTHER:
            current = result.setdefault(name, AttributeValue())
            if cleaned_name == "other" and isinstance(raw_value, Mapping):
                current.merge(AttributeValue.from_raw({"details": raw_value}))
            elif cleaned_name:
                current.merge(
                    AttributeValue(details={cleaned_name: _as_text_list(raw_value)})
                )
            continue

        value = AttributeValue.from_raw(raw_value)
        if value.is_empty():
            continue
        result.setdefault(name, AttributeValue()).merge(value)
    return result


def _merge_product_attribute(
    attributes: AttributeMap,
    name: AttributeName,
    raw_value: object,
) -> None:
    value = AttributeValue.from_raw(raw_value)
    if value.is_empty():
        return
    attributes.setdefault(name, AttributeValue()).merge(value)


def extract_product_attributes(product: Mapping[str, object]) -> AttributeMap:
    """Derive the official ten-field view from one catalog record."""
    attributes: AttributeMap = {}

    raw_categories = _as_text_list(product.get("categories", product.get("category")))
    specific_categories = [
        value for value in raw_categories if value.casefold() not in _GENERIC_CATEGORIES
    ]
    _merge_product_attribute(
        attributes,
        AttributeName.CATEGORY,
        specific_categories or raw_categories,
    )
    _merge_product_attribute(
        attributes,
        AttributeName.FEATURE,
        product.get("features", product.get("feature")),
    )

    price = _as_float(product.get("price"))
    if price is not None:
        attributes[AttributeName.BUDGET] = AttributeValue.range(
            minimum=price,
            maximum=price,
            unit="USD",
        )

    details = product.get("details")
    explicit_brand: list[str] = []
    manufacturer_brand: list[str] = []
    if isinstance(details, Mapping):
        for raw_name, raw_value in details.items():
            cleaned_name = _clean_name(raw_name)
            name = normalize_attribute_name(raw_name)

            if name is AttributeName.BRAND:
                if cleaned_name in {"brand", "brand_name"}:
                    explicit_brand.extend(_as_text_list(raw_value))
                else:
                    manufacturer_brand.extend(_as_text_list(raw_value))
                continue
            if name is AttributeName.BUDGET:
                continue
            if name is AttributeName.SIZE and cleaned_name in _SIZE_DETAIL_ONLY_KEYS:
                _merge_product_attribute(
                    attributes,
                    name,
                    {"details": {cleaned_name: raw_value}},
                )
                continue
            if name in _DETAIL_PRESERVING_FIELDS:
                _merge_product_attribute(
                    attributes,
                    name,
                    {"details": {cleaned_name: raw_value}},
                )
                continue
            _merge_product_attribute(attributes, name, raw_value)

    brand_source: object = explicit_brand or manufacturer_brand
    if not brand_source:
        brand_source = product.get("brand") or product.get("store")
    _merge_product_attribute(attributes, AttributeName.BRAND, brand_source)

    searchable = " ".join(
        _as_text_list(product.get("title"))
        + _as_text_list(product.get("categories"))
        + _as_text_list(product.get("features"))
    )
    for name, pattern in _PRODUCT_TEXT_VALUE_PATTERNS.items():
        if name in attributes:
            continue
        matches = [match.group(0).lower() for match in pattern.finditer(searchable)]
        _merge_product_attribute(attributes, name, matches)

    return attributes


def product_attribute_values(
    attributes: Mapping[AttributeName, AttributeValue],
    name: AttributeName | str,
    *,
    include_details: bool = True,
) -> list[str]:
    """Return deduplicated display/search values for one product attribute."""
    value = attributes.get(normalize_attribute_name(name))
    if value is None:
        return []
    values = list(value.values)
    if include_details:
        for detail_values in value.details.values():
            values.extend(detail_values)
    return _unique(values)


def product_attribute_text(
    attributes: Mapping[AttributeName, AttributeValue],
    name: AttributeName | str,
) -> str:
    """Flatten one product attribute for lexical constraint matching."""
    canonical = normalize_attribute_name(name)
    value = attributes.get(canonical)
    if value is None:
        return ""
    parts = product_attribute_values(attributes, canonical)
    if value.minimum is not None:
        parts.append(str(value.minimum))
    if value.maximum is not None and value.maximum != value.minimum:
        parts.append(str(value.maximum))
    if value.unit:
        parts.append(value.unit)
    return " ".join(_unique(parts))


def attribute_map_to_dict(
    attributes: Mapping[AttributeName, AttributeValue],
) -> dict[str, Any]:
    """Convert an AttributeMap to JSON using only official singular keys."""
    return {
        name.value: value.to_dict()
        for name, value in attributes.items()
        if not value.is_empty()
    }


def to_official_ask_attribute(name: AttributeName | str) -> str:
    """Return an evaluator-valid name; aliases and unknowns are normalized."""
    return normalize_attribute_name(name).value


__all__ = [
    "AttributeName",
    "AttributeValue",
    "AttributeMap",
    "ATTRIBUTE_NAMES",
    "OFFICIAL_ASK_ATTRIBUTES",
    "normalize_attribute_name",
    "normalize_attribute_map",
    "extract_product_attributes",
    "product_attribute_values",
    "product_attribute_text",
    "attribute_map_to_dict",
    "to_official_ask_attribute",
]
