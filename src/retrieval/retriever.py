"""Stable Retrieval facade used by integration and future hybrid strategies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..item import Candidates100
from .bm25 import BM25Retriever, BM25Weights
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
    def hybrid(
        cls,
        bm25: RetrievalStrategy,
        dense: RetrievalStrategy,
        *,
        config: HybridConfig | None = None,
    ) -> Retriever:
        from .hybrid import HybridRetriever

        return cls(HybridRetriever(bm25, dense, config=config))

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        return self.strategy.retrieve(query, state, intent, k)


def retrieve(
    retriever: RetrievalStrategy,
    query: str | None,
    state: object | None = None,
    intent: str | None = None,
    k: int = 100,
) -> Candidates100:
    """Functional adapter preserving the preferred Module 1 call shape."""
    return retriever.retrieve(query, state, intent, k)
