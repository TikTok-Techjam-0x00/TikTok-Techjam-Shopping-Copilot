"""High-precision English slot extraction for shopping turns."""

from __future__ import annotations

import re

from .model import StateUpdate


COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "beige")
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "linen", "suede", "denim", "fabric")
SIZES = ("xxs", "xs", "small", "medium", "large", "xl", "xxl", "wide", "narrow", "petite")
USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work", "wedding", "travel", "daily")

NO_PREFERENCE_RE = re.compile(
    r"(?:don['’]?t|do not) have (?:an? )?(?:additional )?preference for\s+([a-z_]+)|no preference for\s+([a-z_]+)",
    re.I,
)
BUDGET_RE = re.compile(r"(?:under|below|less than|no more than|up to|max(?:imum)?)\s*[$£€]?\s*(\d+(?:\.\d+)?)", re.I)
MONEY_RE = re.compile(r"[$£€]\s*(\d+(?:\.\d+)?)")
CATEGORY_PATTERNS = (
    re.compile(r"(?:i(?:'m| am) looking for|i need|i want|find me|show me)\s+(?:an?\s+)?(.+?)(?=\.|,|;|\bunder\b|\bbelow\b|\bwith\b|\bthat\b|$)", re.I),
    re.compile(r"(?:change that to|instead(?:,)?(?: i need)?|now i need)\s*:?[ ]*(?:an?\s+)?(.+?)(?=\.|,|;|$)", re.I),
)
ATTRIBUTE_ALIASES = {"colour": "color", "price": "budget", "price_max": "budget"}


def _first_term(text: str, values: tuple[str, ...]) -> str | None:
    for value in values:
        if re.search(rf"\b{re.escape(value)}\b", text, re.I):
            return "gray" if value == "grey" else value
    return None


def _extract_category(text: str) -> str | None:
    candidates = [match.group(1) for pattern in CATEGORY_PATTERNS for match in pattern.finditer(text)]
    if not candidates:
        return None
    value = re.sub(r"^(?:some|any|a pair of)\s+", "", candidates[-1].strip(), flags=re.I)
    value = value.strip(" :-;,.")
    return value.lower()[:120] if value else None


def _labeled_requirement(text: str) -> str | None:
    match = re.search(r"(?:a key requirement is|what matters is|what i need is)\s*:\s*(.+)$", text, re.I)
    return match.group(1).strip(" .")[:240] if match else None


def _active_scope(text: str, is_override: bool) -> str:
    if not is_override:
        return text
    markers = list(re.finditer(r"(?:now i need|now i want|what i need is|instead(?:,)?(?: i need)?)\s*:?[ ]*", text, re.I))
    return text[markers[-1].end():] if markers else text


def extract_slots(message: str, *, asked_attribute: str | None = None, is_override: bool = False) -> StateUpdate:
    text = " ".join(message.strip().split())
    if not text:
        return StateUpdate(override=is_override, clear_soft_constraint=is_override)
    active = _active_scope(text, is_override)
    no_preference: set[str] = set()
    raw_hard: dict[str, object] = {}
    raw_soft: dict[str, object] = {}
    raw_rejected: dict[str, object] = {}

    unavailable = NO_PREFERENCE_RE.search(text)
    if unavailable:
        name = next((group for group in unavailable.groups() if group), "other").lower()
        no_preference.add(ATTRIBUTE_ALIASES.get(name, name))
        return StateUpdate.from_raw(no_preference=no_preference, override=is_override, clear_soft_constraint=is_override)

    category = _extract_category(text)
    if category:
        raw_hard["category"] = category
    budget = BUDGET_RE.search(active) or MONEY_RE.search(active)
    if budget:
        raw_hard["budget"] = {"max": float(budget.group(1)), "unit": "USD"}

    for name, value in (
        ("color", _first_term(active, COLORS)),
        ("material", _first_term(active, MATERIALS)),
        ("size", _first_term(active, SIZES)),
        ("use_case", _first_term(active, USE_CASES)),
    ):
        if value:
            raw_hard[name] = value

    brand = re.search(r"\b(?:brand|by)\s*[:=]?\s*([\w&.'-]+(?:\s+[\w&.'-]+){0,2})", active, re.I)
    if brand:
        raw_hard["brand"] = brand.group(1).strip().lower()

    for name, values in (("material", MATERIALS), ("color", COLORS), ("size", SIZES)):
        negative = re.search(rf"\b(?:not|no|avoid|without)\s+({'|'.join(map(re.escape, values))})\b", active, re.I)
        if negative:
            value = negative.group(1).lower()
            raw_rejected[name] = "gray" if value == "grey" else value
            raw_hard.pop(name, None)

    labeled = _labeled_requirement(text)
    if labeled:
        raw_soft["feature"] = labeled
    elif asked_attribute and asked_attribute not in raw_hard:
        answer = re.sub(r"^(?:for that,?\s*)?(?:what matters is\s*:\s*)?", "", text, flags=re.I).strip(" .")
        if answer and len(answer) <= 240:
            target = ATTRIBUTE_ALIASES.get(asked_attribute, asked_attribute)
            if target in {"category", "brand", "size", "material", "color", "budget", "use_case"}:
                raw_hard[target] = answer.lower()
            else:
                raw_soft[target] = answer

    return StateUpdate.from_raw(
        hard_constraint=raw_hard,
        soft_constraint=raw_soft,
        rejected_values=raw_rejected,
        override=is_override,
        clear_soft_constraint=is_override,
    )
