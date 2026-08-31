"""Retrieval facade for BM25 routing and optional semantic residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..item import Candidates100
from .bm25 import BM25Retriever, BM25Weights
from .catalog import Catalog
from .text import DEFAULT_TEXT_VERSION, ProductTextConfig

# Buying keeps the strong identity/constraint columns while reducing noisy
# merchant and long-description matches.  The Browsing warm start favors the
# category hierarchy because its first query is intentionally broad.
SOTA_BUYING_WEIGHTS = BM25Weights(
    title=6.0,
    categories=4.0,
    features=2.5,
    attributes=2.5,
    details=2.5,
    store=0.75,
    description=0.5,
)
SOTA_BROWSING_WEIGHTS = BM25Weights(
    title=5.0,
    categories=6.0,
    features=2.5,
    attributes=2.5,
    details=2.5,
    store=1.5,
    description=1.0,
)


class RetrievalStrategy(Protocol):
    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100: ...


class Retriever:
    """Shared retrieval entry points for the production Agent and frontend."""

    def __init__(self, strategy: RetrievalStrategy) -> None:
        self.strategy = strategy

    def reset_usage(self) -> None:
        """Reset optional provider usage for the next Agent turn."""

        reset = getattr(self.strategy, "reset_usage", None)
        if callable(reset):
            reset()

    def model_usage(self) -> tuple[int, int]:
        """Return optional provider usage since the last reset."""

        usage = getattr(self.strategy, "model_usage", None)
        if callable(usage):
            return usage()
        return 0, 0

    @classmethod
    def bm25(
        cls,
        catalog_path: Catalog | str | Path,
        *,
        weights: BM25Weights | None = None,
        text_version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
    ) -> Retriever:
        return cls(
            BM25Retriever(
                catalog_path,
                weights=weights,
                text_version=text_version,
            )
        )

    @classmethod
    def bm25_intent_routed(
        cls,
        catalog_path: Catalog | str | Path,
        *,
        buying_text_version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
        browsing_text_version: str | ProductTextConfig = "title_category_v1",
        browsing_max_turn: int | None = 1,
        weights: BM25Weights | None = None,
        buying_weights: BM25Weights | None = None,
        browsing_weights: BM25Weights | None = None,
    ) -> Retriever:
        """Use detailed text for Buying and concise title/category for Browsing."""
        from .routing import IntentRoutedRetriever, IntentRoutingConfig

        catalog = catalog_path if isinstance(catalog_path, Catalog) else Catalog.load(catalog_path)
        return cls(
            IntentRoutedRetriever(
                BM25Retriever(
                    catalog,
                    weights=buying_weights or weights,
                    text_version=buying_text_version,
                ),
                BM25Retriever(
                    catalog,
                    weights=browsing_weights or weights,
                    text_version=browsing_text_version,
                ),
                config=IntentRoutingConfig(browsing_max_turn=browsing_max_turn),
            )
        )

    @classmethod
    def sota_default(cls, catalog_path: Catalog | str | Path) -> Retriever:
        """Return the deterministic intent-routed BM25 production strategy."""
        return cls.bm25_intent_routed(
            catalog_path,
            buying_weights=SOTA_BUYING_WEIGHTS,
            browsing_weights=SOTA_BROWSING_WEIGHTS,
        )

    @classmethod
    def sota_semantic_residual(
        cls,
        catalog_path: Catalog | str | Path,
        *,
        cache_root: str | Path = Path("artifacts") / "retrieval" / "dense",
    ) -> Retriever:
        """Add one late semantic exploration cohort to the SOTA BM25 route.

        BM25 remains the primary retriever on every ordinary page.  When the
        precomputed needs-vector cache and embedding credentials are available,
        turn 8 may request a small lexical-gated semantic residual through
        :meth:`retrieve_residual_page`. Missing configuration or an invalid
        cache returns the exact deterministic BM25 strategy.
        """
        from .dense import DenseRetriever
        from .embedding import OpenAIEmbeddingEncoder
        from .residual import LexicalGatedResidualRetriever, ResidualDenseConfig

        lexical = cls.sota_default(catalog_path)
        needs_cache = (
            Path(cache_root)
            / "text-embedding-v4__dense_needs_v1__d256"
        )
        required = ("manifest.json", "parent_asins.json", "embeddings.npy")
        if not all((needs_cache / filename).is_file() for filename in required):
            return lexical
        try:
            encoder = OpenAIEmbeddingEncoder.from_env()
            catalog = getattr(lexical.strategy, "catalog", None)
            if catalog is None:
                catalog = catalog_path if isinstance(catalog_path, Catalog) else Catalog.load(catalog_path)
            semantic = DenseRetriever(
                catalog,
                encoder,
                needs_cache,
                text_version="dense_needs_v1",
                query_embedding_mode="query_instruction",
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return lexical
        return cls(
            LexicalGatedResidualRetriever(
                lexical.strategy,
                semantic,
                config=ResidualDenseConfig(
                    protected_lexical=10,
                    semantic_slots=10,
                    semantic_source_k=200,
                    lexical_gate_depth=1000,
                    minimum_lexical_rank=301,
                    fill_lexical_tail=False,
                ),
            )
        )

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        return self.strategy.retrieve(query, state, intent, k)

    def retrieve_page(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        *,
        page: int = 0,
        page_size: int = 100,
    ) -> Candidates100:
        """Return one deterministic page from a deeper strategy result.

        Later conversation turns can use this to explore beyond an exhausted
        first candidate pool without changing the RetrievalStrategy contract.
        """

        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            raise ValueError("page must be a non-negative integer")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        stop = (page + 1) * page_size
        candidates = self.strategy.retrieve(query, state, intent, stop)
        start = page * page_size
        return candidates[start:stop]

    def retrieve_residual_page(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        *,
        page: int = 0,
        page_size: int = 100,
    ) -> Candidates100:
        """Return a strategy-specific residual page or the ordinary page.

        This keeps the production facade deterministic for BM25 while allowing
        an explicitly configured strategy to add a bounded semantic quota.
        """

        residual = getattr(self.strategy, "retrieve_residual_page", None)
        if callable(residual):
            return residual(
                query,
                state,
                intent,
                page=page,
                page_size=page_size,
            )
        return self.retrieve_page(
            query,
            state,
            intent,
            page=page,
            page_size=page_size,
        )

    def retrieve_strata(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        *,
        windows: tuple[tuple[int, int], ...],
    ) -> Candidates100:
        """Return deterministic rank windows from one deeper retrieval call.

        This supports late-turn breadth without issuing the same dense or BM25
        query once per window. The strict Retrieval contract is preserved by
        limiting the combined pool to at most 100 candidates.
        """

        if not windows:
            return []
        normalized: list[tuple[int, int]] = []
        for start, size in windows:
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
            ):
                raise ValueError("window starts must be non-negative integers")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise ValueError("window sizes must be positive integers")
            normalized.append((start, size))
        if sum(size for _, size in normalized) > 100:
            raise ValueError("combined retrieval windows must not exceed 100")

        stop = max(start + size for start, size in normalized)
        candidates = self.strategy.retrieve(query, state, intent, stop)
        selected: Candidates100 = []
        seen: set[str] = set()
        for start, size in normalized:
            for candidate in candidates[start:start + size]:
                if candidate.parent_asin in seen:
                    continue
                seen.add(candidate.parent_asin)
                selected.append(candidate)
        return selected


def retrieve(
    retriever: RetrievalStrategy,
    query: str | None,
    state: object | None = None,
    intent: str | None = None,
    k: int = 100,
) -> Candidates100:
    """Functional adapter preserving the preferred Module 1 call shape."""
    return retriever.retrieve(query, state, intent, k)
