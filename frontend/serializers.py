"""JSON-safe views of existing pipeline objects for the developer inspector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert existing structured objects without inventing diagnostic data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    return str(value)


def _candidate_fields(candidate: Any, raw: dict[str, Any]) -> dict[str, Any]:
    item = raw.get("item", {}) if isinstance(raw, dict) else {}
    return {
        "parent_asin": item.get("parent_asin", getattr(candidate, "parent_asin", "")),
        "product": item,
        "bm25_score": raw.get("bm25_score"),
        "dense_score": raw.get("dense_score"),
        "retrieval_score": raw.get("retrieval_score"),
        "retrieval_rank": raw.get("retrieval_rank"),
    }


def candidate_view(candidate: Any) -> dict[str, Any]:
    return _candidate_fields(candidate, json_safe(candidate))


def ranked_candidate_view(candidate: Any) -> dict[str, Any]:
    raw = json_safe(candidate)
    return {
        **_candidate_fields(candidate, raw),
        "rerank_score": raw.get("rerank_score"),
        "rerank_rank": raw.get("rerank_rank"),
        "matched": raw.get("matched", []),
        "violation": raw.get("violation", []),
    }
