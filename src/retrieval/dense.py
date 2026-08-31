"""Dense Retrieval over cached, normalized catalog embeddings."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..item import Candidate, Candidates100
from .catalog import Catalog
from .embedding import EmbeddingEncoder, LoadedEmbeddingCache, load_embedding_cache
from .query import build_retrieval_query
from .text import DEFAULT_TEXT_VERSION


QUERY_EMBEDDING_MODES = frozenset({"symmetric", "query", "query_instruction"})


class DenseRetrievalError(RuntimeError):
    """Wrap provider failures so semantic residuals can fall back to BM25."""


class DenseRetriever:
    """Cosine-similarity search over a memory-mapped product matrix.

    Catalog embeddings are L2-normalized while building the cache. Query
    embeddings are normalized here, so a matrix-vector dot product is cosine
    similarity. All exposed scores follow higher-is-better semantics.
    """

    def __init__(
        self,
        catalog: Catalog | str,
        encoder: EmbeddingEncoder,
        cache_dir: str | Path,
        *,
        text_version: str = DEFAULT_TEXT_VERSION,
        query_cache_size: int = 512,
        query_embedding_mode: str = "symmetric",
        query_instruction: str | None = None,
    ) -> None:
        self.catalog = catalog if isinstance(catalog, Catalog) else Catalog.load(catalog)
        self.encoder = encoder
        self.text_version = text_version
        self.query_cache_size = max(1, int(query_cache_size))
        if query_embedding_mode not in QUERY_EMBEDDING_MODES:
            raise ValueError(
                "query_embedding_mode must be one of: "
                + ", ".join(sorted(QUERY_EMBEDDING_MODES))
            )
        self.query_embedding_mode = query_embedding_mode
        self.query_instruction = query_instruction
        self.cache: LoadedEmbeddingCache = load_embedding_cache(
            self.catalog,
            encoder,
            cache_dir,
            text_version=text_version,
        )
        self._query_vectors: OrderedDict[str, np.ndarray] = OrderedDict()

    def _remember_query(self, query: str, vector: np.ndarray) -> None:
        if self.query_cache_size <= 0:
            return
        self._query_vectors[query] = vector
        self._query_vectors.move_to_end(query)
        while len(self._query_vectors) > self.query_cache_size:
            self._query_vectors.popitem(last=False)

    @staticmethod
    def _normalize_query(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("query embedding contains non-finite values")
        norm = float(np.linalg.norm(values))
        return values / norm if norm > 0.0 else values

    def preload_queries(self, queries: Sequence[str], *, batch_size: int | None = None) -> None:
        """Encode and cache missing queries without duplicate provider calls."""
        missing = list(dict.fromkeys(query for query in queries if query and query not in self._query_vectors))
        size = int(batch_size or getattr(self.encoder, "batch_size", 10))
        if size <= 0:
            raise ValueError("batch_size must be positive")
        for offset in range(0, len(missing), size):
            batch = missing[offset : offset + size]
            try:
                if self.query_embedding_mode == "symmetric":
                    raw_vectors = self.encoder.encode(batch)
                else:
                    encode_queries = getattr(self.encoder, "encode_queries", None)
                    if not callable(encode_queries):
                        raise TypeError(
                            "encoder does not support DashScope query embeddings"
                        )
                    instruction = (
                        self.query_instruction
                        if self.query_embedding_mode == "query_instruction"
                        else None
                    )
                    if instruction is None and self.query_embedding_mode == "query_instruction":
                        instruction = getattr(
                            getattr(self.encoder, "config", None),
                            "query_instruction",
                            None,
                        )
                    raw_vectors = encode_queries(batch, instruct=instruction)
                encoded = np.asarray(raw_vectors, dtype=np.float32)
            except Exception as error:
                raise DenseRetrievalError("query embedding provider failed") from error
            expected = (len(batch), self.encoder.dimension)
            if encoded.shape != expected:
                raise DenseRetrievalError(
                    f"query embedding provider returned {encoded.shape}; expected {expected}"
                )
            for query, vector in zip(batch, encoded, strict=True):
                self._remember_query(query, self._normalize_query(vector))

    def _query_vector(self, query: str) -> np.ndarray:
        cached = self._query_vectors.get(query)
        if cached is not None:
            self._query_vectors.move_to_end(query)
            return cached
        self.preload_queries([query], batch_size=1)
        return self._query_vectors[query]

    @staticmethod
    def _best_indices(scores: np.ndarray, k: int) -> np.ndarray:
        count = min(k, len(scores))
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        if count == len(scores):
            return np.argsort(-scores, kind="stable")
        selected = np.argpartition(scores, -count)[-count:]
        return selected[np.lexsort((selected, -scores[selected]))]

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
        vector = self._query_vector(retrieval_query)
        if not np.any(vector):
            return []
        scores = np.asarray(self.cache.embeddings @ vector, dtype=np.float32)
        indices = self._best_indices(scores, k)

        candidates: Candidates100 = []
        for rank, index in enumerate(indices, start=1):
            parent_asin = self.cache.parent_asins[int(index)]
            product = self.catalog[parent_asin]
            score = float(scores[int(index)])
            candidates.append(
                Candidate(
                    item=product,
                    dense_score=score,
                    retrieval_score=score,
                    retrieval_rank=rank,
                )
            )
        return candidates

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> DenseRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["QUERY_EMBEDDING_MODES", "DenseRetrievalError", "DenseRetriever"]
