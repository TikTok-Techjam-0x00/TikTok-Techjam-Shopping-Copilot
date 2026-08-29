"""Stable Retrieval facade used by integration and future hybrid strategies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..item import Candidates100
from .bm25 import BM25Retriever, BM25Weights
from .catalog import Catalog
from .text import DEFAULT_TEXT_VERSION, ProductTextConfig

if TYPE_CHECKING:
    from .embedding import EmbeddingEncoder
    from .hybrid import HybridConfig


class RetrievalStrategy(Protocol):
    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100: ...


class Retriever:
    """Small replaceable facade for lexical, dense, and hybrid strategies."""

    def __init__(self, strategy: RetrievalStrategy) -> None:
        self.strategy = strategy

    @classmethod
    def bm25(
        cls,
        catalog_path: str,
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
    def dense(
        cls,
        catalog_path: str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path,
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        query_embedding_mode: str = "query_instruction",
        query_instruction: str | None = None,
    ) -> Retriever:
        from .dense import DenseRetriever

        return cls(
            DenseRetriever(
                catalog_path,
                encoder,
                cache_dir,
                text_version=text_version,
                query_embedding_mode=query_embedding_mode,
                query_instruction=query_instruction,
            )
        )

    @classmethod
    def bm25_intent_routed(
        cls,
        catalog_path: str,
        *,
        buying_text_version: str | ProductTextConfig = DEFAULT_TEXT_VERSION,
        browsing_text_version: str | ProductTextConfig = "title_category_v1",
        browsing_max_turn: int | None = 1,
        weights: BM25Weights | None = None,
    ) -> Retriever:
        """Use detailed text for Buying and concise title/category for Browsing."""
        from .routing import IntentRoutedRetriever, IntentRoutingConfig

        catalog = Catalog.load(catalog_path)
        return cls(
            IntentRoutedRetriever(
                BM25Retriever(
                    catalog,
                    weights=weights,
                    text_version=buying_text_version,
                ),
                BM25Retriever(
                    catalog,
                    weights=weights,
                    text_version=browsing_text_version,
                ),
                config=IntentRoutingConfig(browsing_max_turn=browsing_max_turn),
            )
        )

    @classmethod
    def sota_default(cls, catalog_path: str) -> Retriever:
        """Return the measured SOTA Hybrid, with a startup BM25 fallback.

        The Hybrid itself also falls back per query when the embedding service
        is temporarily unavailable.  This outer fallback covers missing API
        configuration, dependencies, or product embedding cache at startup.
        """
        try:
            from .embedding import OpenAIEmbeddingEncoder

            return cls.intent_routed_hybrid_weighted(
                catalog_path,
                OpenAIEmbeddingEncoder.from_env(),
            )
        except Exception:
            return cls.bm25_intent_routed(catalog_path)

    @classmethod
    def hybrid(
        cls,
        bm25: RetrievalStrategy,
        dense: RetrievalStrategy,
        *,
        config: HybridConfig | None = None,
    ) -> Retriever:
        from .hybrid import HybridRetriever

        return cls(HybridRetriever(bm25, dense, config=config))

    @classmethod
    def bm25_dense_rrf(
        cls,
        catalog_path: str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path | None = None,
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        source_k: int = 100,
        rank_constant: float = 60.0,
        query_embedding_mode: str = "query_instruction",
        query_instruction: str | None = None,
        fallback_to_bm25: bool = True,
    ) -> Retriever:
        """Build the team-ready BM25 + Dense reciprocal-rank fusion strategy."""
        from .hybrid import HybridConfig

        return cls._bm25_dense(
            catalog_path,
            encoder,
            cache_dir,
            text_version=text_version,
            query_embedding_mode=query_embedding_mode,
            query_instruction=query_instruction,
            config=HybridConfig(
                method="rrf",
                source_k=source_k,
                rank_constant=rank_constant,
                fallback_to_bm25=fallback_to_bm25,
            ),
        )

    @classmethod
    def bm25_dense_weighted(
        cls,
        catalog_path: str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path | None = None,
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        source_k: int = 100,
        alpha: float = 0.5,
        query_embedding_mode: str = "query_instruction",
        query_instruction: str | None = None,
        fallback_to_bm25: bool = True,
    ) -> Retriever:
        """Build the team-ready normalized BM25 + Dense weighted strategy."""
        from .hybrid import HybridConfig

        return cls._bm25_dense(
            catalog_path,
            encoder,
            cache_dir,
            text_version=text_version,
            query_embedding_mode=query_embedding_mode,
            query_instruction=query_instruction,
            config=HybridConfig(
                method="weighted",
                source_k=source_k,
                alpha=alpha,
                fallback_to_bm25=fallback_to_bm25,
            ),
        )

    @classmethod
    def bm25_intent_routed_dense_weighted(
        cls,
        catalog_path: str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path | None = None,
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        browsing_text_version: str | ProductTextConfig = "title_category_v1",
        browsing_max_turn: int | None = 1,
        source_k: int = 200,
        alpha: float = 0.7,
        query_embedding_mode: str = "query_instruction",
        query_instruction: str | None = None,
        fallback_to_bm25: bool = True,
    ) -> Retriever:
        """Build the RET-008 winner: routed BM25 + Dense weighted fusion.

        Buying and later Browsing turns use ``text_version``.  Only the first
        Browsing turn uses the concise text representation.  The defaults are
        the best configuration measured on the public multi-turn Retrieval
        evaluation; Dense failures retain the deterministic BM25 route.
        """
        from .dense import DenseRetriever
        from .embedding import default_embedding_cache_dir
        from .hybrid import HybridConfig, HybridRetriever
        from .routing import IntentRoutedRetriever, IntentRoutingConfig

        catalog = Catalog.load(catalog_path)
        resolved_cache = cache_dir or default_embedding_cache_dir(
            encoder.model,
            encoder.dimension,
            text_version,
        )
        bm25 = IntentRoutedRetriever(
            BM25Retriever(catalog, text_version=text_version),
            BM25Retriever(catalog, text_version=browsing_text_version),
            config=IntentRoutingConfig(browsing_max_turn=browsing_max_turn),
        )
        dense = DenseRetriever(
            catalog,
            encoder,
            resolved_cache,
            text_version=text_version,
            query_embedding_mode=query_embedding_mode,
            query_instruction=query_instruction,
        )
        return cls(
            HybridRetriever(
                bm25,
                dense,
                config=HybridConfig(
                    method="weighted",
                    source_k=source_k,
                    alpha=alpha,
                    fallback_to_bm25=fallback_to_bm25,
                ),
            )
        )

    @classmethod
    def intent_routed_hybrid_weighted(
        cls,
        catalog_path: str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path | None = None,
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        browsing_text_version: str | ProductTextConfig = "title_category_v1",
        browsing_max_turn: int | None = 1,
        source_k: int = 200,
        alpha: float = 0.7,
        query_embedding_mode: str = "query_instruction",
        query_instruction: str | None = None,
        fallback_to_bm25: bool = True,
    ) -> Retriever:
        """Route Browsing warm-start outside the detailed Hybrid strategy.

        The first Browsing turn uses concise BM25 alone, matching the measured
        SOTA lexical policy without reintroducing long-field Dense noise. Buying,
        unknown intent, and later Browsing turns use detailed BM25 + Dense.
        """
        from .dense import DenseRetriever
        from .embedding import default_embedding_cache_dir
        from .hybrid import HybridConfig, HybridRetriever
        from .routing import IntentRoutedRetriever, IntentRoutingConfig

        catalog = Catalog.load(catalog_path)
        resolved_cache = cache_dir or default_embedding_cache_dir(
            encoder.model,
            encoder.dimension,
            text_version,
        )
        detailed_bm25 = BM25Retriever(catalog, text_version=text_version)
        browsing_bm25 = BM25Retriever(catalog, text_version=browsing_text_version)
        dense = DenseRetriever(
            catalog,
            encoder,
            resolved_cache,
            text_version=text_version,
            query_embedding_mode=query_embedding_mode,
            query_instruction=query_instruction,
        )
        detailed_hybrid = HybridRetriever(
            detailed_bm25,
            dense,
            config=HybridConfig(
                method="weighted",
                source_k=source_k,
                alpha=alpha,
                fallback_to_bm25=fallback_to_bm25,
            ),
        )
        return cls(
            IntentRoutedRetriever(
                detailed_hybrid,
                browsing_bm25,
                config=IntentRoutingConfig(browsing_max_turn=browsing_max_turn),
            )
        )

    @classmethod
    def _bm25_dense(
        cls,
        catalog_path: str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path | None,
        *,
        text_version: str,
        query_embedding_mode: str,
        query_instruction: str | None,
        config: HybridConfig,
    ) -> Retriever:
        from .dense import DenseRetriever
        from .embedding import default_embedding_cache_dir
        from .hybrid import HybridRetriever

        catalog = Catalog.load(catalog_path)
        resolved_cache = cache_dir or default_embedding_cache_dir(
            encoder.model,
            encoder.dimension,
            text_version,
        )
        bm25 = BM25Retriever(catalog, text_version=text_version)
        dense = DenseRetriever(
            catalog,
            encoder,
            resolved_cache,
            text_version=text_version,
            query_embedding_mode=query_embedding_mode,
            query_instruction=query_instruction,
        )
        return cls(HybridRetriever(bm25, dense, config=config))

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


def retrieve(
    retriever: RetrievalStrategy,
    query: str | None,
    state: object | None = None,
    intent: str | None = None,
    k: int = 100,
) -> Candidates100:
    """Functional adapter preserving the preferred Module 1 call shape."""
    return retriever.retrieve(query, state, intent, k)
