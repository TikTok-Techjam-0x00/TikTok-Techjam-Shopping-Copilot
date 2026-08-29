"""Construct lexical queries from Module 2's current shopping state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


_CONSTRAINT_FIELDS = ("hard_constraint", "soft_constraint")
_DIRECT_STATE_FIELDS = ("category", "use_case", "preferences", "user_preferences")
_RANGE_KEYS = frozenset({"min", "minimum", "max", "maximum", "unit", "currency"})
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _retrieval_safe(value: object) -> str:
    return " ".join(CJK_RE.sub(" ", str(value or "")).split())


def _state_value(state: object, field: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(field, default)
    return getattr(state, field, default)


def _attribute_name(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().casefold()


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _provenance_record(
    state: object,
    group: str,
    attribute: object,
) -> object | None:
    provenance = _state_value(state, "constraint_provenance", {})
    if not isinstance(provenance, Mapping):
        return None
    records = provenance.get(group, {})
    if not isinstance(records, Mapping):
        return None
    name = _attribute_name(attribute)
    return records.get(attribute, records.get(name))


def _record_value(record: object, field: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


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
    values: list[tuple[int, list[str]]] = []
    current_epoch = _integer(_state_value(state, "constraint_epoch", 0))
    last_override_turn = _integer(
        _state_value(state, "last_override_turn", -1),
        -1,
    )

    for field in _DIRECT_STATE_FIELDS:
        if field not in excluded:
            direct_values = _lexical_values(_state_value(state, field))
            if direct_values:
                values.append((0, direct_values))

    for field in _CONSTRAINT_FIELDS:
        constraints = _state_value(state, field)
        if not isinstance(constraints, Mapping):
            continue
        for attribute, constraint in constraints.items():
            name = _attribute_name(attribute)
            if name in excluded:
                continue
            lexical = _lexical_values(constraint)
            if not lexical:
                continue
            record = _provenance_record(state, field, attribute)
            record_epoch = _integer(
                _record_value(record, "constraint_epoch", current_epoch),
                current_epoch,
            )
            source_turn = _integer(_record_value(record, "source_turn", -1), -1)
            if name == "category":
                priority = 0
            elif source_turn == last_override_turn and last_override_turn >= 0:
                priority = 1
            elif record_epoch == current_epoch:
                priority = 2
            elif field == "soft_constraint" and current_epoch > 0:
                # Stale soft preferences remain in State for diagnostics and 3A,
                # but do not compete equally in an override retrieval query.
                continue
            else:
                priority = 3
            values.append((priority, lexical))

    result: list[str] = []
    seen: set[str] = set()
    for _, group_values in sorted(values, key=lambda entry: entry[0]):
        for value in group_values:
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
            return _retrieval_safe(" ".join(state_terms))
    raw_query = query
    if (raw_query is None or not str(raw_query).strip()) and state is not None:
        raw_query = _state_value(state, "user_message", "")
    return _retrieval_safe(raw_query)
