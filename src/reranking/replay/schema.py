"""Stable JSON contracts for reranking replay datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...item import Candidate, Item


SCHEMA_VERSION = "reranking-replay-v1"


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value)


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    """Compact frozen Retrieval output; the Item is restored from the catalog."""

    parent_asin: str
    bm25_score: float | None = None
    dense_score: float | None = None
    retrieval_score: float | None = None
    retrieval_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.parent_asin.strip():
            raise ValueError("ReplayCandidate.parent_asin must not be empty")

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> ReplayCandidate:
        return cls(
            parent_asin=candidate.parent_asin,
            bm25_score=candidate.bm25_score,
            dense_score=candidate.dense_score,
            retrieval_score=candidate.retrieval_score,
            retrieval_rank=candidate.retrieval_rank,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayCandidate:
        return cls(
            parent_asin=str(value.get("parent_asin") or ""),
            bm25_score=_optional_float(value.get("bm25_score")),
            dense_score=_optional_float(value.get("dense_score")),
            retrieval_score=_optional_float(value.get("retrieval_score")),
            retrieval_rank=_optional_int(value.get("retrieval_rank")),
        )

    def to_candidate(self, item: Item) -> Candidate:
        if item.parent_asin != self.parent_asin:
            raise ValueError("catalog Item does not match ReplayCandidate.parent_asin")
        return Candidate(
            item=item,
            bm25_score=self.bm25_score,
            dense_score=self.dense_score,
            retrieval_score=self.retrieval_score,
            retrieval_rank=self.retrieval_rank,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
        }


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """One exact pre-reranking input captured from one evaluator turn."""

    case_id: str
    sample_id: str
    scenario_type: str
    turn: int
    scorable: bool
    override_applied: bool
    shopping_state: dict[str, Any]
    candidates_100: tuple[ReplayCandidate, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.sample_id:
            raise ValueError("ReplayCase identifiers must not be empty")
        if self.turn <= 0:
            raise ValueError("ReplayCase.turn must be positive")
        if len(self.candidates_100) > 100:
            raise ValueError("ReplayCase cannot contain more than 100 candidates")
        identifiers = [candidate.parent_asin for candidate in self.candidates_100]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ReplayCase candidates must have unique parent_asin values")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayCase:
        raw_state = value.get("shopping_state")
        raw_candidates = value.get("candidates_100")
        if not isinstance(raw_state, Mapping):
            raise ValueError("shopping_state must be an object")
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates_100 must be a list")
        return cls(
            case_id=str(value.get("case_id") or ""),
            sample_id=str(value.get("sample_id") or ""),
            scenario_type=str(value.get("scenario_type") or "unknown"),
            turn=int(value.get("turn") or 0),
            scorable=bool(value.get("scorable", True)),
            override_applied=bool(value.get("override_applied", True)),
            shopping_state=dict(raw_state),
            candidates_100=tuple(
                ReplayCandidate.from_dict(candidate)
                for candidate in raw_candidates
                if isinstance(candidate, Mapping)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "scenario_type": self.scenario_type,
            "turn": self.turn,
            "scorable": self.scorable,
            "override_applied": self.override_applied,
            "shopping_state": dict(self.shopping_state),
            "candidates_100": [candidate.to_dict() for candidate in self.candidates_100],
        }


@dataclass(frozen=True, slots=True)
class ReplayLabel:
    """Ground truth stored separately and consumed only after ranking."""

    case_id: str
    target_parent_asin: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.target_parent_asin:
            raise ValueError("ReplayLabel values must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayLabel:
        return cls(
            case_id=str(value.get("case_id") or ""),
            target_parent_asin=str(value.get("target_parent_asin") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "target_parent_asin": self.target_parent_asin,
        }
