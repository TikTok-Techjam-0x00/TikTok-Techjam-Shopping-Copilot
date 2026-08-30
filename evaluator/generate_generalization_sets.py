"""Build frozen, public-target-disjoint generalization sets for local evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from evaluator.local_evaluator import behavior_for, intent_card, load_jsonl


SET_SIZE = 800
SCENARIO_COUNTS = {
    "browsing": 320,
    "buying": 320,
    "intent_override": 120,
    "boundary": 40,
}
SEEDS = {
    "iid": 2026082901,
    "long_tail": 2026082902,
    "stress": 2026082903,
}
DIMENSION_RE = re.compile(
    r"\b(?:up to\s+)?\d+(?:\.\d+)?\s*(?:mm|cm|m|inches?|in\b|feet|foot|ft)\b",
    re.I,
)
WIDTH_RE = re.compile(r"\b(?:wide|narrow|width|diameter|dimension|circumference)\b", re.I)
MONEY_LIKE_RE = re.compile(r"\b(?:money|value|saving|cost|price|budget)\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leaf_category(product: dict) -> str:
    categories = [str(value).strip().casefold() for value in product.get("categories") or []]
    return categories[-1] if categories else "uncategorized"


def _searchable_fragments(product: dict) -> list[str]:
    fragments = [str(value) for value in product.get("features") or []]
    details = product.get("details")
    if isinstance(details, dict):
        fragments.extend(f"{key}: {value}" for key, value in details.items())
    fragments.extend(str(value) for value in product.get("description") or [])
    return [" ".join(fragment.split()) for fragment in fragments if str(fragment).strip()]


def _stress_score(product: dict) -> tuple[int, int]:
    text = " ".join(_searchable_fragments(product))
    score = 0
    score += 8 * len(DIMENSION_RE.findall(text))
    score += 4 * len(re.findall(r"\bup to\b", text, re.I))
    score += 3 * len(WIDTH_RE.findall(text))
    score += 2 * len(MONEY_LIKE_RE.findall(text))
    score += min(4, len(NUMBER_RE.findall(text)) // 3)
    if product.get("price") in (None, ""):
        score += 1
    return score, len(text)


def _round_robin_groups(
    groups: dict[str, list[dict]],
    ordered_keys: list[str],
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in ordered_keys:
            values = groups[key]
            if depth >= len(values):
                continue
            selected.append(values[depth])
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        depth += 1
    if len(selected) != limit:
        raise RuntimeError(f"could only select {len(selected)} of {limit} products")
    return selected


def _select_iid(products: list[dict], rng: random.Random) -> list[dict]:
    return rng.sample(products, SET_SIZE)


def _select_long_tail(products: list[dict], rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        groups[_leaf_category(product)].append(product)
    for values in groups.values():
        rng.shuffle(values)
        values.sort(key=lambda item: (int(item.get("rating_number") or 0), str(item["parent_asin"])))
    keys = sorted(groups, key=lambda key: (len(groups[key]), key))
    return _round_robin_groups(groups, keys, SET_SIZE)


def _select_stress(products: list[dict], rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        if _stress_score(product)[0] > 0:
            groups[_leaf_category(product)].append(product)
    for values in groups.values():
        rng.shuffle(values)
        values.sort(key=lambda item: (_stress_score(item), str(item["parent_asin"])), reverse=True)
    keys = sorted(
        groups,
        key=lambda key: (_stress_score(groups[key][0]), -len(groups[key]), key),
        reverse=True,
    )
    return _round_robin_groups(groups, keys, SET_SIZE)


def _stress_requirement(product: dict) -> str | None:
    fragments = _searchable_fragments(product)
    dimension = [value for value in fragments if DIMENSION_RE.search(value)]
    if dimension:
        return max(dimension, key=lambda value: (_stress_score({"features": [value]})[0], len(value)))[:180]
    noisy = [value for value in fragments if WIDTH_RE.search(value) or MONEY_LIKE_RE.search(value)]
    return max(noisy, key=len)[:180] if noisy else None


def _profile(product: dict, index: int) -> dict:
    tags = ["material", "fit", "color", "style", "durability", "comfort"]
    first = tags[index % len(tags)]
    second = tags[(index * 3 + 1) % len(tags)]
    rating = float(product.get("average_rating") or 4.0)
    return {
        "average_prior_rating": max(1.0, min(5.0, rating)),
        "preference_tags": list(dict.fromkeys((first, second))),
        "purchase_frequency": ("1-2 prior purchases", "3-4 prior purchases", "5+ prior purchases")[index % 3],
        "rating_style": ("mixed", "usually positive")[index % 2],
        "summary": f"Held-out synthetic profile emphasizing {first} and {second}.",
    }


def _scenario_sequence(rng: random.Random) -> list[str]:
    scenarios = [name for name, count in SCENARIO_COUNTS.items() for _ in range(count)]
    rng.shuffle(scenarios)
    return scenarios


def _materialize_rows(name: str, products: list[dict], rng: random.Random) -> list[dict]:
    scenarios = _scenario_sequence(rng)
    rows: list[dict] = []
    for index, (product, scenario) in enumerate(zip(products, scenarios), start=1):
        card = intent_card(product)
        if name == "stress":
            requirement = _stress_requirement(product)
            if requirement:
                existing = [
                    *[str(value) for value in card.get("hard_constraints", [])],
                    *[str(value) for value in card.get("soft_preferences", [])],
                ]
                remaining = [value for value in existing if value.casefold() != requirement.casefold()]
                card = {
                    **card,
                    "hard_constraints": [requirement, *remaining[:1]],
                    "soft_preferences": remaining[1:3] or remaining[:1],
                }
        sample_id = f"gen_{name}_{index:04d}"
        rows.append({
            "sample_id": sample_id,
            "scenario_type": scenario,
            "category_bucket": _leaf_category(product),
            "difficulty_bucket": name,
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
            "user_profile": _profile(product, index),
            "intent_card": card,
            "behavior": behavior_for(scenario, card, random.Random(f"{sample_id}\0{scenario}")),
        })
    return rows


def generate(catalog_path: Path, public_path: Path, output_dir: Path, source_git: str) -> dict:
    catalog = load_jsonl(catalog_path)
    public_targets = {
        str(row["ground_truth"]["parent_asin"])
        for row in load_jsonl(public_path)
    }
    eligible = [
        product for product in catalog
        if str(product.get("parent_asin") or "") not in public_targets
        and product.get("title")
        and product.get("categories")
        and (product.get("features") or product.get("details") or product.get("description"))
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    selectors = {
        "iid": _select_iid,
        "long_tail": _select_long_tail,
        "stress": _select_stress,
    }
    manifest = {
        "source_git": source_git,
        "catalog_sha256": _sha256(catalog_path),
        "public_set_sha256": _sha256(public_path),
        "public_targets_excluded": len(public_targets),
        "set_size": SET_SIZE,
        "scenario_counts": SCENARIO_COUNTS,
        "sets": {},
    }
    selected_asins: dict[str, set[str]] = {}
    remaining = list(eligible)
    for name, selector in selectors.items():
        rng = random.Random(SEEDS[name])
        # Freeze mutually disjoint target sets so one product cannot reward the
        # same implementation choice in more than one generalization slice.
        selected = selector(remaining, rng)
        rows = _materialize_rows(name, selected, rng)
        path = output_dir / f"{name}_800.jsonl"
        _jsonl(path, rows)
        asins = {str(row["ground_truth"]["parent_asin"]) for row in rows}
        selected_asins[name] = asins
        remaining = [
            product
            for product in remaining
            if str(product.get("parent_asin") or "") not in asins
        ]
        manifest["sets"][name] = {
            "seed": SEEDS[name],
            "path": path.name,
            "sha256": _sha256(path),
            "sample_count": len(rows),
            "unique_targets": len(asins),
            "scenario_counts": dict(Counter(row["scenario_type"] for row in rows)),
        }
    manifest["cross_set_target_overlap"] = {
        f"{left}__{right}": len(selected_asins[left] & selected_asins[right])
        for index, left in enumerate(selectors)
        for right in list(selectors)[index + 1:]
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--public-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-git", default="unknown")
    args = parser.parse_args()
    manifest = generate(args.catalog, args.public_set, args.output_dir, args.source_git)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
