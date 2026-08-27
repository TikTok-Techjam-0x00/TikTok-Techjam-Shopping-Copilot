"""Construct lexical queries from Module 2's current shopping state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_CONSTRAINT_FIELDS = ("hard_constraint", "soft_constraint")
_DIRECT_STATE_FIELDS = ("category", "use_case", "preferences", "user_preferences")
_RANGE_KEYS = frozenset({"min", "minimum", "max", "maximum", "unit", "currency"})


def _state_value(state: object, field: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(field, default)
    return getattr(state, field, default)


def _attribute_name(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().casefold()


def _no_preference(state: object) -> set[str]:
    value = _state_value(
        state,
        "no_prefernce",
        _state_value(state, "no_preference", ()),
    )
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ()
    return {_attribute_name(entry) for entry in values}


def _lexical_values(value: object) -> list[str]:
    """Flatten text values while omitting range-only metadata such as a budget."""
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return []
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        return [cleaned] if cleaned else []

    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key, entry in value.items():
            if _attribute_name(key) in _RANGE_KEYS:
                continue
            flattened.extend(_lexical_values(entry))
        return flattened

    object_values = getattr(value, "values", None)
    object_details = getattr(value, "details", None)
    if object_values is not None or object_details is not None:
        return [
            *_lexical_values(object_values),
            *_lexical_values(object_details),
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened = []
        for entry in value:
            flattened.extend(_lexical_values(entry))
        return flattened
    return _lexical_values(str(value))


def _active_state_terms(state: object) -> list[str]:
    excluded = _no_preference(state)
    values: list[str] = []

    for field in _DIRECT_STATE_FIELDS:
        if field not in excluded:
            values.extend(_lexical_values(_state_value(state, field)))

    for field in _CONSTRAINT_FIELDS:
        constraints = _state_value(state, field)
        if not isinstance(constraints, Mapping):
            continue
        for attribute, constraint in constraints.items():
            if _attribute_name(attribute) not in excluded:
                values.extend(_lexical_values(constraint))

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def build_retrieval_query(
    query: str | None,
    state: object | None = None,
    intent: str | None = None,
) -> str:
    """Return an explicit, testable lexical query.

    When Module 2 supplies active category/constraint values, they are treated as
    authoritative. This prevents an overridden preference in the raw conversation
    text from leaking back into retrieval. Until state has useful values, the
    current user query (or ``state.user_message``) is used as a fallback.

    ``intent`` is accepted for the stable Retrieval API. The BM25 baseline does
    not narrow buying/browsing differently until an experiment supports it.
    """
    del intent
    if state is not None:
        state_terms = _active_state_terms(state)
        if state_terms:
            return " ".join(state_terms)
    raw_query = query
    if (raw_query is None or not str(raw_query).strip()) and state is not None:
        raw_query = _state_value(state, "user_message", "")
    return " ".join(str(raw_query or "").split())
