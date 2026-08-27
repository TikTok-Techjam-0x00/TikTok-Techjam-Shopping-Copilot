"""Stable Retrieval facade used by integration and future hybrid strategies."""

from __future__ import annotations

from typing import Protocol

from ..item import Candidates100
from .bm25 import BM25Retriever


class RetrievalStrategy(Protocol):
    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100: ...


class Retriever:
    """Small replaceable facade; the current deterministic strategy is BM25."""

    def __init__(self, strategy: RetrievalStrategy) -> None:
        self.strategy = strategy

    @classmethod
    def bm25(cls, catalog_path: str) -> Retriever:
        return cls(BM25Retriever(catalog_path))

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
