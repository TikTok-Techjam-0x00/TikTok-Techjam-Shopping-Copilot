"""Module 1: lexical, dense, and hybrid product retrieval."""

from .bm25 import BM25Retriever, BM25Weights
from .catalog import Catalog, CatalogLoadStats
from .dense import QUERY_EMBEDDING_MODES, DenseRetrievalError, DenseRetriever
from .embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    EmbeddingCacheError,
    EmbeddingEncoder,
    LoadedEmbeddingCache,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingEncoder,
    build_embedding_cache,
    default_embedding_cache_dir,
    load_embedding_cache,
)
from .hybrid import (
    HybridConfig,
    HybridRetriever,
    candidate_union,
    min_max_normalize,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from .multivector import MultiVectorConfig, MultiVectorDenseRetriever
from .query import build_retrieval_query
from .retriever import Retriever, retrieve
from .routing import IntentRoutedRetriever, IntentRoutingConfig
from .text import (
    DEFAULT_TEXT_VERSION,
    TEXT_CONFIGS,
    ProductTextConfig,
    build_bm25_fields,
    build_product_text,
    resolve_text_config,
)

__all__ = [
    "Catalog",
    "CatalogLoadStats",
    "BM25Weights",
    "BM25Retriever",
    "DenseRetriever",
    "DenseRetrievalError",
    "QUERY_EMBEDDING_MODES",
    "DEFAULT_QUERY_INSTRUCTION",
    "EmbeddingEncoder",
    "EmbeddingCacheError",
    "LoadedEmbeddingCache",
    "OpenAIEmbeddingConfig",
    "OpenAIEmbeddingEncoder",
    "build_embedding_cache",
    "load_embedding_cache",
    "default_embedding_cache_dir",
    "HybridConfig",
    "HybridRetriever",
    "candidate_union",
    "min_max_normalize",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "MultiVectorConfig",
    "MultiVectorDenseRetriever",
    "Retriever",
    "IntentRoutedRetriever",
    "IntentRoutingConfig",
    "build_retrieval_query",
    "retrieve",
    "ProductTextConfig",
    "TEXT_CONFIGS",
    "DEFAULT_TEXT_VERSION",
    "resolve_text_config",
    "build_bm25_fields",
    "build_product_text",
]
