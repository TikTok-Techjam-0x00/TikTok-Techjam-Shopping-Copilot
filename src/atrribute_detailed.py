"""Legacy detailed attribute contract kept for experiments and comparison.

This is the richer pre-simplification version. New pipeline code should import
``src.attribute`` instead. Use `AttributeMap` for `shopping_state.hard_constraint` and `soft_constraint`.
Known names are represented by `AttributeName`; unknown source fields are retained
under `AttributeName.OTHERS` instead of being discarded.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


class AttributeName(str, Enum):
    """Stable attribute names used across the shopping pipeline."""

    CATEGORY = "category"
    MATERIAL = "material"
    COLOR = "color"
    SIZE = "size"
    FIT = "fit"
    STYLE = "style"
    PATTERN = "pattern"
    BRAND = "brand"
    BUDGET = "budget"
    FEATURE = "feature"
    USE_CASE = "use_case"
    TARGET_USER = "target_user"
    RATING = "rating"
    QUANTITY = "quantity"
    OTHERS = "others"

    def __str__(self) -> str:
        return self.value


ATTRIBUTE_NAMES: tuple[AttributeName, ...] = tuple(AttributeName)

# Only these names can be returned directly as the official `ask_attribute`.
OFFICIAL_ASK_ATTRIBUTES = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)


_ATTRIBUTE_ALIASES: dict[str, AttributeName] = {
    # Catalog hierarchy / product kind.
    "category": AttributeName.CATEGORY,
    "categories": AttributeName.CATEGORY,
    "product_category": AttributeName.CATEGORY,
    "product_type": AttributeName.CATEGORY,
    "item_type": AttributeName.CATEGORY,
    # Materials.
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
    # Apparel/shoe size plus physical measurements from catalog details.
    "size": AttributeName.SIZE,
    "sizes": AttributeName.SIZE,
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
    # Fit.
    "fit": AttributeName.FIT,
    "fit_type": AttributeName.FIT,
    "shoe_width": AttributeName.FIT,
    # Style / visual form.
    "style": AttributeName.STYLE,
    "theme": AttributeName.STYLE,
    "shape": AttributeName.STYLE,
    "finish_type": AttributeName.STYLE,
    "neck_style": AttributeName.STYLE,
    "sleeve_type": AttributeName.STYLE,
    "heel_type": AttributeName.STYLE,
    # Pattern.
    "pattern": AttributeName.PATTERN,
    "pattern_type": AttributeName.PATTERN,
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
    # Product capabilities and construction details.
    "feature": AttributeName.FEATURE,
    "features": AttributeName.FEATURE,
    "special_feature": AttributeName.FEATURE,
    "special_features": AttributeName.FEATURE,
    "closure_type": AttributeName.FEATURE,
    "included_components": AttributeName.FEATURE,
    # Activity, occasion, or intended situation.
    "use_case": AttributeName.USE_CASE,
    "usecase": AttributeName.USE_CASE,
    "occasion": AttributeName.USE_CASE,
    "sport": AttributeName.USE_CASE,
    "sport_type": AttributeName.USE_CASE,
    "activity": AttributeName.USE_CASE,
    "purpose": AttributeName.USE_CASE,
    # Audience / department / age.
    "target_user": AttributeName.TARGET_USER,
    "target_audience": AttributeName.TARGET_USER,
    "audience": AttributeName.TARGET_USER,
    "department": AttributeName.TARGET_USER,
    "gender": AttributeName.TARGET_USER,
    "age": AttributeName.TARGET_USER,
    "age_range": AttributeName.TARGET_USER,
    "age_range_description": AttributeName.TARGET_USER,
    "suggested_users": AttributeName.TARGET_USER,
    # Rating and popularity requirements.
    "rating": AttributeName.RATING,
    "average_rating": AttributeName.RATING,
    "rating_number": AttributeName.RATING,
    "review_count": AttributeName.RATING,
    # Multipacks and item counts.
    "quantity": AttributeName.QUANTITY,
    "number_of_items": AttributeName.QUANTITY,
    "item_package_quantity": AttributeName.QUANTITY,
    "pack_size": AttributeName.QUANTITY,
    # Explicit fallback.
    "other": AttributeName.OTHERS,
    "others": AttributeName.OTHERS,
}


# Conservative text fallbacks for product fields that are not explicitly
# structured in ``details``.  These are deliberately small: a false product
# attribute can hurt Retrieval filtering, 3A matching, and 3B questions at once.
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
    AttributeName.FIT: re.compile(
        r"\b(slim fit|regular fit|relaxed fit|loose fit|wide|narrow)\b",
        re.IGNORECASE,
    ),
    AttributeName.STYLE: re.compile(
        r"\b(casual|formal|classic|modern|vintage|sporty|slim|relaxed)\b",
        re.IGNORECASE,
    ),
    AttributeName.PATTERN: re.compile(
        r"\b(solid|striped|plaid|floral|polka dot|geometric|animal print)\b",
        re.IGNORECASE,
    ),
    AttributeName.USE_CASE: re.compile(
        r"\b(hiking|running|gym|winter|outdoor|work|wedding|travel|daily)\b",
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
    """One normalized attribute value supporting text, ranges, and subfields.

    Examples:
    - color: `values=["black"]`
    - budget: `minimum=50, maximum=100, unit="USD"`
    - size: `values=["M"], details={"waist": ["32 inches"]}`
    - others: `details={"care_instructions": ["hand wash"]}`
    """

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
        """Normalize common LLM/dataset outputs into one stable value shape."""
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
        """Stable JSON shape; `min`/`max` also work with the current reranker."""
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
    """Map dataset/LLM aliases to a canonical name; unknown names become OTHERS."""
    if isinstance(value, AttributeName):
        return value
    return _ATTRIBUTE_ALIASES.get(_clean_name(value), AttributeName.OTHERS)


def normalize_attribute_map(
    raw: Mapping[str | AttributeName, object] | None,
) -> AttributeMap:
    """Normalize hard/soft constraints and preserve unknown fields in `others`.

    Unknown input such as `{"care_instructions": "hand wash"}` becomes:
    `AttributeName.OTHERS -> details["care_instructions"] = ["hand wash"]`.
    """
    result: AttributeMap = {}
    if not isinstance(raw, Mapping):
        return result

    for raw_name, raw_value in raw.items():
        cleaned_name = _clean_name(raw_name)
        name = normalize_attribute_name(raw_name)
        if name is AttributeName.OTHERS:
            current = result.setdefault(name, AttributeValue())
            if cleaned_name in {"other", "others"} and isinstance(raw_value, Mapping):
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
    """Derive canonical product attributes from one raw catalog record.

    Official fields remain the source of truth.  This function creates an
    internal, shared view for Retrieval, Reranking, and Dialogue without
    inventing new catalog ground truth.  Unknown ``details`` keys are left in
    the original ``details`` mapping instead of being duplicated under OTHERS.
    """
    attributes: AttributeMap = {}

    raw_categories = _as_text_list(
        product.get("categories", product.get("category"))
    )
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
            if name is AttributeName.OTHERS:
                continue
            if name is AttributeName.BRAND:
                if cleaned_name in {"brand", "brand_name"}:
                    explicit_brand.extend(_as_text_list(raw_value))
                else:
                    manufacturer_brand.extend(_as_text_list(raw_value))
                continue
            # The official top-level fields above are more reliable than a
            # similarly named details entry and must not be replaced.
            if name in {AttributeName.BUDGET, AttributeName.RATING}:
                continue
            if name is AttributeName.SIZE and cleaned_name in _SIZE_DETAIL_ONLY_KEYS:
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

    rating = _as_float(product.get("average_rating"))
    rating_number = _as_float(product.get("rating_number"))
    if rating is not None or rating_number is not None:
        rating_details: dict[str, list[str]] = {}
        if rating_number is not None:
            rating_details["rating_number"] = [str(int(rating_number))]
        attributes[AttributeName.RATING] = AttributeValue(
            minimum=rating,
            maximum=rating,
            unit="stars" if rating is not None else None,
            details=rating_details,
        )

    # Explicit catalog metadata wins.  Text inference only fills attributes
    # that are still missing, using title/category/features rather than the
    # much noisier long description.
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
    """Flatten one derived product attribute for lexical constraint matching."""
    value = attributes.get(normalize_attribute_name(name))
    if value is None:
        return ""
    parts = product_attribute_values(attributes, name)
    if value.minimum is not None:
        parts.append(str(value.minimum))
    if value.maximum is not None and value.maximum != value.minimum:
        parts.append(str(value.maximum))
    if value.unit:
        parts.append(value.unit)
    return " ".join(_unique(parts))


def attribute_map_to_dict(attributes: Mapping[AttributeName, AttributeValue]) -> dict[str, Any]:
    """Convert an AttributeMap to a JSON-safe mapping with string keys."""
    return {
        name.value: value.to_dict()
        for name, value in attributes.items()
        if not value.is_empty()
    }


def to_official_ask_attribute(name: AttributeName | str) -> str:
    """Map internal attributes to the evaluator's allowed ask_attribute values."""
    canonical = normalize_attribute_name(name)
    if canonical.value in OFFICIAL_ASK_ATTRIBUTES:
        return canonical.value
    return {
        AttributeName.FIT: "size",
        AttributeName.PATTERN: "style",
    }.get(canonical, "other")


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
