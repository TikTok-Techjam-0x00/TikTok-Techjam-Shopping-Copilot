"""Two-vector product retrieval separating identity from shopping needs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..item import Candidate, Candidates100
from .catalog import Catalog
from .dense import DenseRetriever
from .embedding import EmbeddingEncoder, LoadedEmbeddingCache, load_embedding_cache
from .query import build_retrieval_query


MultiVectorFusion = Literal["weighted", "max"]


@dataclass(frozen=True, slots=True)
class MultiVectorConfig:
    """Configuration for combining identity and needs cosine similarities."""

    fusion: MultiVectorFusion = "weighted"
    identity_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.fusion not in ("weighted", "max"):
            raise ValueError("fusion must be 'weighted' or 'max'")
        if not math.isfinite(self.identity_weight) or not 0.0 <= self.identity_weight <= 1.0:
            raise ValueError("identity_weight must be between 0 and 1")


class MultiVectorDenseRetriever:
    """Search identity and needs embeddings without mixing them into one vector.

    The same instructed query vector is compared with both product matrices. A
    weighted score requires agreement across product identity and constraints;
    max fusion favors recall when either representation is a strong match.
    """

    def __init__(
        self,
        catalog: Catalog | str,
        encoder: EmbeddingEncoder,
        identity_cache_dir: str | Path,
        needs_cache_dir: str | Path,
        *,
        identity_text_version: str = "dense_identity_v1",
        needs_text_version: str = "dense_needs_v1",
        query_embedding_mode: str = "query_instruction",
        query_instruction: str | None = None,
        config: MultiVectorConfig | None = None,
    ) -> None:
        self.catalog = catalog if isinstance(catalog, Catalog) else Catalog.load(catalog)
        self.config = config or MultiVectorConfig()
        # Reuse DenseRetriever's tested query encoding/cache behavior.
        self.identity = DenseRetriever(
            self.catalog,
            encoder,
            identity_cache_dir,
            text_version=identity_text_version,
            query_embedding_mode=query_embedding_mode,
            query_instruction=query_instruction,
        )
        try:
            self.needs: LoadedEmbeddingCache = load_embedding_cache(
                self.catalog,
                encoder,
                needs_cache_dir,
                text_version=needs_text_version,
            )
        except Exception:
            self.identity.close()
            raise

    def preload_queries(
        self,
        queries: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> None:
        self.identity.preload_queries(queries, batch_size=batch_size)

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            return []
        retrieval_query = build_retrieval_query(query, state, intent)
        if not retrieval_query:
            return []
        vector = self.identity._query_vector(retrieval_query)
        if not np.any(vector):
            return []

        identity_scores = np.asarray(
            self.identity.cache.embeddings @ vector,
            dtype=np.float32,
        )
        needs_scores = np.asarray(self.needs.embeddings @ vector, dtype=np.float32)
        if self.config.fusion == "max":
            scores = np.maximum(identity_scores, needs_scores)
        else:
            weight = self.config.identity_weight
            scores = weight * identity_scores + (1.0 - weight) * needs_scores
        indices = DenseRetriever._best_indices(scores, k)

        result: Candidates100 = []
        for rank, index in enumerate(indices, start=1):
            parent_asin = self.identity.cache.parent_asins[int(index)]
            score = float(scores[int(index)])
            result.append(
                Candidate(
                    item=self.catalog[parent_asin],
                    dense_score=score,
                    retrieval_score=score,
                    retrieval_rank=rank,
                )
            )
        return result

    def close(self) -> None:
        self.needs.close()
        self.identity.close()

    def __enter__(self) -> MultiVectorDenseRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["MultiVectorFusion", "MultiVectorConfig", "MultiVectorDenseRetriever"]
