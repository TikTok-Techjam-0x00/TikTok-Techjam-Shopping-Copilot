from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


OFFICIAL_FIELDS = (
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
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|denim|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|beige|navy)\b",
    re.IGNORECASE,
)


def is_nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean")}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "min": round(ordered[0], 4),
        "p10": round(percentile(0.10), 4),
        "p25": round(percentile(0.25), 4),
        "median": round(percentile(0.50), 4),
        "p75": round(percentile(0.75), 4),
        "p90": round(percentile(0.90), 4),
        "p95": round(percentile(0.95), 4),
        "p99": round(percentile(0.99), 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def top(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def analyze(catalog_path: Path) -> dict[str, Any]:
    rows = 0
    invalid_json = 0
    field_present: Counter[str] = Counter()
    field_nonempty: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = {field: Counter() for field in OFFICIAL_FIELDS}
    identifiers: set[str] = set()
    duplicate_identifiers = 0
    title_counts: Counter[str] = Counter()
    stores: Counter[str] = Counter()
    all_categories: Counter[str] = Counter()
    leaf_categories: Counter[str] = Counter()
    detail_keys: Counter[str] = Counter()
    category_depths: list[float] = []
    prices: list[float] = []
    ratings: list[float] = []
    rating_numbers: list[float] = []
    search_chars: list[float] = []
    search_tokens: list[float] = []
    empty_search_documents = 0
    material_coverage = 0
    color_coverage = 0
    price_buckets: Counter[str] = Counter()

    with catalog_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                product = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if not isinstance(product, dict):
                invalid_json += 1
                continue

            rows += 1
            for field in OFFICIAL_FIELDS:
                if field in product:
                    field_present[field] += 1
                value = product.get(field)
                field_types[field][type(value).__name__] += 1
                if is_nonempty(value):
                    field_nonempty[field] += 1

            parent_asin = str(product.get("parent_asin") or "").strip()
            if parent_asin in identifiers:
                duplicate_identifiers += 1
            elif parent_asin:
                identifiers.add(parent_asin)

            title = str(product.get("title") or "").strip()
            if title:
                title_counts[title.casefold()] += 1

            store = str(product.get("store") or "").strip()
            if store:
                stores[store] += 1

            categories = product.get("categories") or []
            if isinstance(categories, list):
                normalized_categories = [str(value).strip() for value in categories if str(value).strip()]
                category_depths.append(float(len(normalized_categories)))
                all_categories.update(normalized_categories)
                if normalized_categories:
                    leaf_categories[normalized_categories[-1]] += 1

            details = product.get("details") or {}
            if isinstance(details, dict):
                detail_keys.update(str(key) for key, value in details.items() if is_nonempty(value))

            price = numeric(product.get("price"))
            if price is None:
                price_buckets["missing_or_invalid"] += 1
            else:
                prices.append(price)
                if price <= 25:
                    price_buckets["0-25"] += 1
                elif price <= 50:
                    price_buckets["25-50"] += 1
                elif price <= 100:
                    price_buckets["50-100"] += 1
                elif price <= 200:
                    price_buckets["100-200"] += 1
                else:
                    price_buckets[">200"] += 1

            rating = numeric(product.get("average_rating"))
            if rating is not None:
                ratings.append(rating)
            rating_number = numeric(product.get("rating_number"))
            if rating_number is not None:
                rating_numbers.append(rating_number)

            searchable_text = " ".join(flatten_text(product.get(field)) for field in SEARCH_FIELDS).strip()
            if not searchable_text:
                empty_search_documents += 1
            else:
                search_chars.append(float(len(searchable_text)))
                search_tokens.append(float(len(TOKEN_RE.findall(searchable_text))))
                material_coverage += int(bool(MATERIAL_RE.search(searchable_text)))
                color_coverage += int(bool(COLOR_RE.search(searchable_text)))

    duplicate_title_groups = sum(1 for count in title_counts.values() if count > 1)
    duplicate_title_items = sum(count for count in title_counts.values() if count > 1)

    def coverage(count: int) -> dict[str, Any]:
        return {
            "count": count,
            "percent": round(100 * count / rows, 2) if rows else 0.0,
        }

    return {
        "source": str(catalog_path),
        "rows": rows,
        "invalid_json_rows": invalid_json,
        "unique_parent_asin": len(identifiers),
        "duplicate_parent_asin": duplicate_identifiers,
        "field_coverage": {
            field: {
                "present": field_present[field],
                "nonempty": field_nonempty[field],
                "nonempty_percent": round(100 * field_nonempty[field] / rows, 2) if rows else 0.0,
                "types": dict(field_types[field].most_common()),
            }
            for field in OFFICIAL_FIELDS
        },
        "price": {
            "valid": len(prices),
            "coverage_percent": round(100 * len(prices) / rows, 2) if rows else 0.0,
            "distribution": quantiles(prices),
            "buckets": dict(price_buckets),
        },
        "average_rating": {
            "valid": len(ratings),
            "coverage_percent": round(100 * len(ratings) / rows, 2) if rows else 0.0,
            "distribution": quantiles(ratings),
        },
        "rating_number": {
            "valid": len(rating_numbers),
            "coverage_percent": round(100 * len(rating_numbers) / rows, 2) if rows else 0.0,
            "distribution": quantiles(rating_numbers),
        },
        "search_text": {
            "empty_documents": empty_search_documents,
            "characters": quantiles(search_chars),
            "approximate_tokens": quantiles(search_tokens),
            "material_keyword_coverage": coverage(material_coverage),
            "color_keyword_coverage": coverage(color_coverage),
        },
        "categories": {
            "depth": quantiles(category_depths),
            "unique_values": len(all_categories),
            "unique_leaf_values": len(leaf_categories),
            "top_all": top(all_categories),
            "top_leaf": top(leaf_categories),
        },
        "stores": {
            "unique": len(stores),
            "top": top(stores),
        },
        "details": {
            "unique_keys": len(detail_keys),
            "top_keys": top(detail_keys, limit=30),
        },
        "titles": {
            "unique_normalized": len(title_counts),
            "duplicate_groups": duplicate_title_groups,
            "items_in_duplicate_groups": duplicate_title_items,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen TechJam product catalog.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/catalog_analysis.json"))
    args = parser.parse_args()

    result = analyze(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
