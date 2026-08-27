from __future__ import annotations

import re
from typing import Any

from starter.state import StateUpdate


COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "beige",
)
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "linen", "suede", "denim", "fabric",
)
SIZES = ("xxs", "xs", "small", "medium", "large", "xl", "xxl", "wide", "narrow", "petite")
USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work", "wedding", "travel", "daily")

NO_PREFERENCE_RE = re.compile(
    r"(?:don['’]?t|do not) have (?:an? )?(?:additional )?preference for\s+([a-z_]+)|"
    r"no preference for\s+([a-z_]+)",
    re.IGNORECASE,
)
BUDGET_RE = re.compile(
    r"(?:under|below|less than|no more than|up to|max(?:imum)?)"
    r"\s*[$£€¥]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"[$£€¥]\s*(\d+(?:\.\d+)?)")
CATEGORY_PATTERNS = (
    re.compile(r"(?:i(?:'m| am) looking for|i need|i want|find me|show me)\s+(?:an?\s+)?(.+?)(?=\.|,|;|\bunder\b|\bbelow\b|\bwith\b|\bthat\b|$)", re.I),
    re.compile(r"(?:change that to|instead(?:,)?(?: i need)?|now i need)\s*:?[ ]*(?:an?\s+)?(.+?)(?=\.|,|;|$)", re.I),
)

ATTRIBUTE_ALIASES = {
    "colour": "color",
    "price": "budget",
    "price_max": "budget",
}


def _first_term(text: str, values: tuple[str, ...]) -> str | None:
    for value in values:
        if re.search(rf"\b{re.escape(value)}\b", text, re.I):
            return "gray" if value == "grey" else value
    return None


def _clean_category(value: str) -> str | None:
    value = re.sub(r"^(?:some|any|a pair of)\s+", "", value.strip(), flags=re.I)
    value = re.sub(r"\s+(?:but i(?:'m| am) still exploring|please)$", "", value, flags=re.I)
    value = value.strip(" :-;,.")
    return value.lower()[:120] if value else None


def _extract_category(text: str) -> str | None:
    # During an override the final need is authoritative, so inspect matches in
    # reverse order and prefer the last category phrase.
    candidates: list[str] = []
    for pattern in CATEGORY_PATTERNS:
        candidates.extend(match.group(1) for match in pattern.finditer(text))
    return _clean_category(candidates[-1]) if candidates else None


def _extract_labeled_requirement(text: str) -> str | None:
    match = re.search(
        r"(?:a key requirement is|what matters is|what i need is)\s*:\s*(.+)$",
        text,
        re.I,
    )
    return match.group(1).strip(" .")[:240] if match else None


def _active_scope(text: str, is_override: bool) -> str:
    """Ignore discarded values before the final override clause."""

    if not is_override:
        return text
    markers = list(re.finditer(
        r"(?:now i need|now i want|what i need is|instead(?:,)?(?: i need)?)\s*:?[ ]*",
        text,
        re.I,
    ))
    return text[markers[-1].end():] if markers else text


def extract_slots(
    message: str,
    *,
    asked_attribute: str | None = None,
    is_override: bool = False,
) -> StateUpdate:
    """Extract deterministic slots from one English shopping turn.

    The parser deliberately prefers precision over guessing. Unstructured
    evaluator requirements are retained as ``feature`` text so retrieval does
    not lose useful catalog wording.
    """

    text = " ".join(message.strip().split())
    update = StateUpdate(override=is_override, clear_soft_preferences=is_override)
    if not text:
        return update

    active_text = _active_scope(text, is_override)

    no_preference = NO_PREFERENCE_RE.search(text)
    if no_preference:
        raw = next((group for group in no_preference.groups() if group), "other").lower()
        update.rejected_attributes.add(ATTRIBUTE_ALIASES.get(raw, raw))
        return update

    category = _extract_category(text)
    if category:
        update.category = category
        update.clear_soft_preferences = is_override

    budget = BUDGET_RE.search(active_text) or MONEY_RE.search(active_text)
    if budget:
        update.hard_constraints["budget"] = float(budget.group(1))

    color = _first_term(active_text, COLORS)
    material = _first_term(active_text, MATERIALS)
    size = _first_term(active_text, SIZES)
    use_case = _first_term(active_text, USE_CASES)
    for attribute, value in (
        ("color", color),
        ("material", material),
        ("size", size),
        ("use_case", use_case),
    ):
        if value:
            update.hard_constraints[attribute] = value

    brand = re.search(r"\b(?:brand|by)\s*[:=]?\s*([\w&.'-]+(?:\s+[\w&.'-]+){0,2})", active_text, re.I)
    if brand:
        update.hard_constraints["brand"] = brand.group(1).strip().lower()

    for attribute, values in (("material", MATERIALS), ("color", COLORS), ("size", SIZES)):
        negative = re.search(
            rf"\b(?:not|no|avoid|without)\s+({'|'.join(map(re.escape, values))})\b",
            active_text,
            re.I,
        )
        if negative:
            value = negative.group(1).lower()
            update.negative_constraints[attribute] = "gray" if value == "grey" else value
            update.hard_constraints.pop(attribute, None)

    labeled = _extract_labeled_requirement(text)
    if labeled:
        # Preserve the exact descriptive constraint even when a structured slot
        # was also found; BM25 can use both signals.
        if labeled not in update.hard_constraints.values():
            update.soft_preferences.append(labeled)
    elif asked_attribute and asked_attribute not in update.hard_constraints:
        # A direct answer to the previous question is meaningful even if its
        # vocabulary is not in the small rule dictionary.
        answer = re.sub(r"^(?:for that,?\s*)?(?:what matters is\s*:\s*)?", "", text, flags=re.I)
        answer = answer.strip(" .")
        if answer and len(answer) <= 240:
            if asked_attribute in {"category", "brand", "size", "material", "color", "budget", "use_case"}:
                update.hard_constraints.setdefault(asked_attribute, answer.lower())
            else:
                update.soft_preferences.append(answer)

    return update
