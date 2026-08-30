"""Conservative semantic supplementation for strong lexical retrieval.

Unlike score fusion, this strategy never lets a dense-only result displace the
lexical head.  Dense retrieval may fill a small residual quota only when the
candidate is independently supported by a deeper BM25 result.  Provider errors
fall back to the original lexical page byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..item import Candidate, Candidates100
from .dense import DenseRetrievalError
from .retriever import RetrievalStrategy


@dataclass(frozen=True, slots=True)
class ResidualDenseConfig:
    """Parameters for a bounded, lexical-gated semantic residual."""

    protected_lexical: int = 80
    semantic_slots: int = 20
    semantic_source_k: int = 100
    lexical_gate_depth: int = 1000
    minimum_lexical_rank: int = 301
    fill_lexical_tail: bool = True

    def __post_init__(self) -> None:
        values = (
            self.protected_lexical,
            self.semantic_slots,
            self.semantic_source_k,
            self.lexical_gate_depth,
            self.minimum_lexical_rank,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("residual retrieval parameters must be integers")
        if self.protected_lexical < 0:
            raise ValueError("protected_lexical must not be negative")
        if self.semantic_slots < 0:
            raise ValueError("semantic_slots must not be negative")
        if self.semantic_source_k <= 0:
            raise ValueError("semantic_source_k must be positive")
        if self.lexical_gate_depth <= 0:
            raise ValueError("lexical_gate_depth must be positive")
        if self.minimum_lexical_rank <= 0:
            raise ValueError("minimum_lexical_rank must be positive")
        if not isinstance(self.fill_lexical_tail, bool):
            raise TypeError("fill_lexical_tail must be a boolean")


class LexicalGatedResidualRetriever:
    """Keep BM25 as the primary retriever and use Dense only for residual slots.

    ``retrieve`` is intentionally identical to the lexical strategy.  The
    semantic path is opt-in through ``retrieve_residual_page`` so early turns
    and the established BM25 exploration schedule remain unchanged.
    """

    def __init__(
        self,
        lexical: RetrievalStrategy,
        semantic: RetrievalStrategy,
        *,
        config: ResidualDenseConfig | None = None,
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.config = config or ResidualDenseConfig()
        self.catalog = getattr(lexical, "catalog", None)

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        return self.lexical.retrieve(query, state, intent, k)

    def retrieve_residual_page(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        *,
        page: int,
        page_size: int,
    ) -> Candidates100:
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError("page must be a non-negative integer")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
            raise ValueError("page_size must be a positive integer")

        start = page * page_size
        stop = start + page_size
        depth = max(stop, self.config.lexical_gate_depth)
        lexical = self.lexical.retrieve(query, state, intent, depth)
        base_page = lexical[start:stop]
        if not base_page or self.config.semantic_slots == 0:
            return base_page

        protected_count = min(
            len(base_page),
            self.config.protected_lexical,
            max(0, page_size - self.config.semantic_slots),
        )
        protected = list(base_page[:protected_count])
        lexical_by_id = {
            candidate.parent_asin: (rank, candidate)
            for rank, candidate in enumerate(lexical, start=1)
        }

        try:
            semantic = self.semantic.retrieve(
                query,
                state,
                intent,
                self.config.semantic_source_k,
            )
        except (DenseRetrievalError, RuntimeError, OSError, ValueError):
            return base_page

        selected: Candidates100 = list(protected)
        seen = {candidate.parent_asin for candidate in selected}
        supplements = 0
        for dense_rank, dense_candidate in enumerate(semantic, start=1):
            lexical_match = lexical_by_id.get(dense_candidate.parent_asin)
            if lexical_match is None:
                continue
            lexical_rank, bm25_candidate = lexical_match
            if lexical_rank < self.config.minimum_lexical_rank:
                continue
            if dense_candidate.parent_asin in seen:
                continue
            # Reciprocal-rank consensus is bounded and comparable across score
            # distributions.  It is metadata only; the semantic quota and
            # lexical gate, not a fragile score scale, determine membership.
            consensus = (
                1.0 / (60.0 + lexical_rank)
                + 1.0 / (60.0 + dense_rank)
            )
            selected.append(
                Candidate(
                    item=bm25_candidate.item,
                    bm25_score=bm25_candidate.bm25_score,
                    dense_score=dense_candidate.dense_score,
                    retrieval_score=consensus,
                    retrieval_rank=len(selected) + 1,
                )
            )
            seen.add(dense_candidate.parent_asin)
            supplements += 1
            if supplements >= self.config.semantic_slots:
                break

        if self.config.fill_lexical_tail:
            for candidate in base_page[protected_count:]:
                if candidate.parent_asin in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.parent_asin)
                if len(selected) >= page_size:
                    break
        return selected[:page_size]


__all__ = ["ResidualDenseConfig", "LexicalGatedResidualRetriever"]
