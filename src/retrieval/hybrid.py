"""BM25 + Dense candidate union and rank/score fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..item import Candidate, Candidates100
from .dense import DenseRetrievalError
from .retriever import RetrievalStrategy


FusionMethod = Literal["rrf", "weighted"]


def _positive_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    return k


def _candidate_copy(
    source: Candidate,
    *,
    bm25_score: float | None = None,
    dense_score: float | None = None,
    retrieval_score: float | None = None,
    retrieval_rank: int | None = None,
) -> Candidate:
    return Candidate(
        item=source.item,
        bm25_score=source.bm25_score if bm25_score is None else bm25_score,
        dense_score=source.dense_score if dense_score is None else dense_score,
        retrieval_score=retrieval_score,
        retrieval_rank=retrieval_rank,
    )


def candidate_union(
    bm25_candidates: Sequence[Candidate],
    dense_candidates: Sequence[Candidate],
) -> Candidates100:
    """Return BM25 Top-K union Dense Top-K with exact ASIN deduplication.

    This is a candidate pool and can contain up to ``2K`` products. Its order is
    deterministic (BM25 first, then Dense-only products), but that order is not
    a cross-retriever score. RRF/weighted fusion should be used for a strict
    ranked Top-K result.
    """
    merged: dict[str, Candidate] = {}
    for candidate in bm25_candidates:
        if candidate.parent_asin not in merged:
            merged[candidate.parent_asin] = _candidate_copy(
                candidate,
                retrieval_score=None,
            )
    for candidate in dense_candidates:
        existing = merged.get(candidate.parent_asin)
        if existing is None:
            merged[candidate.parent_asin] = _candidate_copy(
                candidate,
                retrieval_score=None,
            )
        else:
            existing.dense_score = candidate.dense_score

    result: Candidates100 = []
    for rank, candidate in enumerate(merged.values(), start=1):
        candidate.retrieval_rank = rank
        result.append(candidate)
    return result


def _rank_by_asin(candidates: Sequence[Candidate]) -> dict[str, int]:
    return {
        candidate.parent_asin: rank
        for rank, candidate in enumerate(candidates, start=1)
    }


def _raw_scores(
    candidates: Sequence[Candidate],
    field: Literal["bm25_score", "dense_score"],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in candidates:
        value = getattr(candidate, field)
        if value is None:
            value = candidate.retrieval_score
        if value is not None and math.isfinite(float(value)):
            scores[candidate.parent_asin] = float(value)
    return scores


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize one retriever's scores to [0, 1], higher remaining better."""
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    span = maximum - minimum
    if span <= 1e-12:
        return {parent_asin: 1.0 for parent_asin in scores}
    return {
        parent_asin: (score - minimum) / span
        for parent_asin, score in scores.items()
    }


def _sorted_fusion(
    pool: Sequence[Candidate],
    scores: dict[str, float],
    bm25_ranks: dict[str, int],
    dense_ranks: dict[str, int],
    k: int,
) -> Candidates100:
    ordered = sorted(
        pool,
        key=lambda candidate: (
            -scores[candidate.parent_asin],
            min(
                bm25_ranks.get(candidate.parent_asin, math.inf),
                dense_ranks.get(candidate.parent_asin, math.inf),
            ),
            candidate.parent_asin,
        ),
    )[:k]
    return [
        _candidate_copy(
            candidate,
            retrieval_score=scores[candidate.parent_asin],
            retrieval_rank=rank,
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]


def reciprocal_rank_fusion(
    bm25_candidates: Sequence[Candidate],
    dense_candidates: Sequence[Candidate],
    *,
    k: int = 100,
    rank_constant: float = 60.0,
) -> Candidates100:
    """Fuse BM25 and Dense ranks without comparing incompatible raw scores."""
    _positive_k(k)
    if k <= 0:
        return []
    if not math.isfinite(rank_constant) or rank_constant < 0:
        raise ValueError("rank_constant must be finite and non-negative")
    pool = candidate_union(bm25_candidates, dense_candidates)
    bm25_ranks = _rank_by_asin(bm25_candidates)
    dense_ranks = _rank_by_asin(dense_candidates)
    scores: dict[str, float] = {}
    for candidate in pool:
        score = 0.0
        if candidate.parent_asin in bm25_ranks:
            score += 1.0 / (rank_constant + bm25_ranks[candidate.parent_asin])
        if candidate.parent_asin in dense_ranks:
            score += 1.0 / (rank_constant + dense_ranks[candidate.parent_asin])
        scores[candidate.parent_asin] = score
    return _sorted_fusion(pool, scores, bm25_ranks, dense_ranks, k)


def weighted_score_fusion(
    bm25_candidates: Sequence[Candidate],
    dense_candidates: Sequence[Candidate],
    *,
    k: int = 100,
    alpha: float = 0.5,
) -> Candidates100:
    """Fuse independently min-max-normalized BM25 and Dense scores."""
    _positive_k(k)
    if k <= 0:
        return []
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    pool = candidate_union(bm25_candidates, dense_candidates)
    bm25_ranks = _rank_by_asin(bm25_candidates)
    dense_ranks = _rank_by_asin(dense_candidates)
    bm25_scores = min_max_normalize(_raw_scores(bm25_candidates, "bm25_score"))
    dense_scores = min_max_normalize(_raw_scores(dense_candidates, "dense_score"))
    scores = {
        candidate.parent_asin: (
            alpha * bm25_scores.get(candidate.parent_asin, 0.0)
            + (1.0 - alpha) * dense_scores.get(candidate.parent_asin, 0.0)
        )
        for candidate in pool
    }
    return _sorted_fusion(pool, scores, bm25_ranks, dense_ranks, k)


@dataclass(frozen=True, slots=True)
class HybridConfig:
    method: FusionMethod = "rrf"
    source_k: int = 100
    rank_constant: float = 60.0
    alpha: float = 0.5
    fallback_to_bm25: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("rrf", "weighted"):
            raise ValueError("method must be 'rrf' or 'weighted'")
        if self.source_k <= 0:
            raise ValueError("source_k must be positive")


class HybridRetriever:
    """Retrieve from BM25 and Dense, then return a fused strict Top-K."""

    def __init__(
        self,
        bm25: RetrievalStrategy,
        dense: RetrievalStrategy,
        *,
        config: HybridConfig | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.config = config or HybridConfig()

    def retrieve_sources(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        *,
        k: int | None = None,
    ) -> tuple[Candidates100, Candidates100]:
        source_k = max(self.config.source_k, k or 0)
        bm25_candidates = self.bm25.retrieve(query, state, intent, source_k)
        try:
            dense_candidates = self.dense.retrieve(query, state, intent, source_k)
        except DenseRetrievalError:
            if not self.config.fallback_to_bm25:
                raise
            dense_candidates = []
        return bm25_candidates, dense_candidates

    def retrieve_pool(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        *,
        k_per_source: int | None = None,
    ) -> Candidates100:
        bm25_candidates, dense_candidates = self.retrieve_sources(
            query,
            state,
            intent,
            k=k_per_source,
        )
        return candidate_union(bm25_candidates, dense_candidates)

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        _positive_k(k)
        if k <= 0:
            return []
        bm25_candidates, dense_candidates = self.retrieve_sources(
            query,
            state,
            intent,
            k=k,
        )
        if not dense_candidates:
            return [
                _candidate_copy(
                    candidate,
                    retrieval_score=candidate.retrieval_score,
                    retrieval_rank=rank,
                )
                for rank, candidate in enumerate(bm25_candidates[:k], start=1)
            ]
        if self.config.method == "rrf":
            return reciprocal_rank_fusion(
                bm25_candidates,
                dense_candidates,
                k=k,
                rank_constant=self.config.rank_constant,
            )
        return weighted_score_fusion(
            bm25_candidates,
            dense_candidates,
            k=k,
            alpha=self.config.alpha,
        )


__all__ = [
    "FusionMethod",
    "HybridConfig",
    "HybridRetriever",
    "candidate_union",
    "min_max_normalize",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
