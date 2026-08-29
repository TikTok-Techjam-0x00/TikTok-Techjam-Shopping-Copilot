"""Embedding providers and reproducible on-disk catalog caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from dotenv import load_dotenv

from .catalog import Catalog
from .text import DEFAULT_TEXT_VERSION, build_product_text, resolve_text_config


CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_ROOT = Path("artifacts") / "retrieval" / "dense"
DEFAULT_QUERY_INSTRUCTION = (
    "Given the current shopping request, retrieve catalog products that best match "
    "the requested product type and disclosed constraints. Prioritize the product "
    "type and hard requirements; treat soft preferences as secondary signals."
)


class EmbeddingEncoder(Protocol):
    """Small provider boundary used by cache building and query retrieval."""

    model: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class EmbeddingCacheError(RuntimeError):
    """Raised when a cache is absent, stale, malformed, or incompatible."""


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingConfig:
    """Configuration for an OpenAI-compatible embedding endpoint."""

    api_key: str
    base_url: str
    model: str = "text-embedding-v4"
    dimension: int = 256
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 10
    max_chars: int = 12_000
    dashscope_base_url: str = ""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    @classmethod
    def from_env(cls) -> OpenAIEmbeddingConfig:
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        base_url = os.getenv("QWEN_BASE_URL", "").strip()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Dense Retrieval")
        if not base_url:
            raise ValueError("QWEN_BASE_URL is required for Dense Retrieval")
        dashscope_base_url = os.getenv("QWEN_DASHSCOPE_BASE_URL", "").strip()
        if not dashscope_base_url:
            dashscope_base_url = _dashscope_base_url(base_url)
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4").strip(),
            dimension=_positive_int(os.getenv("QWEN_EMBEDDING_DIM", "256"), "QWEN_EMBEDDING_DIM"),
            timeout_seconds=_positive_float(
                os.getenv("QWEN_API_TIMEOUT_SECONDS", "30"),
                "QWEN_API_TIMEOUT_SECONDS",
            ),
            max_retries=_non_negative_int(
                os.getenv("QWEN_API_MAX_RETRIES", "3"),
                "QWEN_API_MAX_RETRIES",
            ),
            batch_size=_positive_int(
                os.getenv("QWEN_EMBEDDING_BATCH_SIZE", "10"),
                "QWEN_EMBEDDING_BATCH_SIZE",
            ),
            max_chars=_positive_int(
                os.getenv("QWEN_EMBEDDING_MAX_CHARS", "12000"),
                "QWEN_EMBEDDING_MAX_CHARS",
            ),
            dashscope_base_url=dashscope_base_url,
            query_instruction=os.getenv(
                "QWEN_QUERY_INSTRUCTION",
                DEFAULT_QUERY_INSTRUCTION,
            ).strip(),
        )


class OpenAIEmbeddingEncoder:
    """Encode text through the project's Qwen/OpenAI-compatible endpoint."""

    def __init__(self, config: OpenAIEmbeddingConfig) -> None:
        from openai import OpenAI

        self.config = config
        self.model = config.model
        self.dimension = config.dimension
        self.batch_size = config.batch_size
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    @classmethod
    def from_env(cls) -> OpenAIEmbeddingEncoder:
        return cls(OpenAIEmbeddingConfig.from_env())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prepared = [str(text or " ")[: self.config.max_chars] for text in texts]
        response = self._client.embeddings.create(
            model=self.model,
            input=prepared,
            dimensions=self.dimension,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda value: value.index)
        matrix = np.asarray([value.embedding for value in ordered], dtype=np.float32)
        _validate_embedding_batch(matrix, len(prepared), self.dimension)
        return matrix

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        instruct: str | None = None,
    ) -> np.ndarray:
        """Encode search queries through DashScope's asymmetric query mode."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prepared = [str(text or " ")[: self.config.max_chars] for text in texts]
        parameters: dict[str, Any] = {
            "dimension": self.dimension,
            "output_type": "dense",
            "text_type": "query",
        }
        if instruct:
            parameters["instruct"] = instruct
        payload = json.dumps(
            {
                "model": self.model,
                "input": {"texts": prepared},
                "parameters": parameters,
            }
        ).encode("utf-8")
        endpoint = (
            self.config.dashscope_base_url.rstrip("/")
            + "/services/embeddings/text-embedding/text-embedding"
        )
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                embeddings = body.get("output", {}).get("embeddings", [])
                ordered = sorted(
                    embeddings,
                    key=lambda value: int(value.get("text_index", value.get("index", 0))),
                )
                matrix = np.asarray(
                    [value["embedding"] for value in ordered],
                    dtype=np.float32,
                )
                _validate_embedding_batch(matrix, len(prepared), self.dimension)
                return matrix
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            if attempt < self.config.max_retries:
                time.sleep(min(2**attempt, 4))
        raise RuntimeError("DashScope query embedding request failed") from last_error


@dataclass(frozen=True, slots=True)
class EmbeddingCacheManifest:
    format_version: int
    model: str
    dimension: int
    text_version: str
    catalog_fingerprint: str
    item_count: int
    normalized: bool
    created_at_utc: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EmbeddingCacheManifest:
        try:
            return cls(
                format_version=int(value["format_version"]),
                model=str(value["model"]),
                dimension=int(value["dimension"]),
                text_version=str(value["text_version"]),
                catalog_fingerprint=str(value["catalog_fingerprint"]),
                item_count=int(value["item_count"]),
                normalized=bool(value["normalized"]),
                created_at_utc=str(value["created_at_utc"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingCacheError("embedding cache manifest is malformed") from error


@dataclass(slots=True)
class LoadedEmbeddingCache:
    cache_dir: Path
    manifest: EmbeddingCacheManifest
    parent_asins: tuple[str, ...]
    embeddings: np.ndarray
    cache_hit: bool

    def close(self) -> None:
        """Release an ``np.memmap`` promptly, which is required on Windows."""
        memory_map = getattr(self.embeddings, "_mmap", None)
        if memory_map is not None:
            memory_map.close()


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _dashscope_base_url(compatible_base_url: str) -> str:
    """Derive the native DashScope URL from the configured compatible URL."""
    cleaned = compatible_base_url.rstrip("/")
    suffix = "/compatible-mode/v1"
    if cleaned.endswith(suffix):
        return cleaned[: -len(suffix)] + "/api/v1"
    raise ValueError(
        "QWEN_DASHSCOPE_BASE_URL is required when QWEN_BASE_URL does not end "
        "with /compatible-mode/v1"
    )


def _non_negative_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_embedding_batch(matrix: np.ndarray, rows: int, dimension: int) -> None:
    if matrix.shape != (rows, dimension):
        raise ValueError(
            f"embedding provider returned shape {matrix.shape}; expected {(rows, dimension)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embedding provider returned non-finite values")


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    """Return float32 L2-normalized rows, preserving zero vectors."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return np.asarray(values / safe_norms, dtype=np.float32)


def catalog_text_fingerprint(
    catalog: Catalog,
    text_version: str = DEFAULT_TEXT_VERSION,
) -> str:
    """Hash product identity and the exact text sent to the embedding model."""
    config = resolve_text_config(text_version)
    digest = hashlib.sha256()
    digest.update(f"text_version={config.name}\n".encode("utf-8"))
    for product in catalog.items_in_order:
        digest.update(product.parent_asin.encode("utf-8"))
        digest.update(b"\0")
        digest.update(build_product_text(product, config).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def default_embedding_cache_dir(
    model: str,
    dimension: int,
    text_version: str = DEFAULT_TEXT_VERSION,
    *,
    root: str | Path = DEFAULT_CACHE_ROOT,
) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("._") or "model"
    config = resolve_text_config(text_version)
    return Path(root) / f"{safe_model}__{config.name}__d{dimension}"


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows Defender/indexers can briefly hold the destination file.
            # Retrying preserves the atomic checkpoint without weakening it to
            # an in-place write that could leave truncated JSON after a crash.
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingCacheError(f"cannot read embedding cache file: {path}") from error


def load_embedding_cache(
    catalog: Catalog,
    encoder: EmbeddingEncoder,
    cache_dir: str | Path,
    *,
    text_version: str = DEFAULT_TEXT_VERSION,
    fingerprint: str | None = None,
) -> LoadedEmbeddingCache:
    """Validate and memory-map a complete embedding cache."""
    directory = Path(cache_dir)
    manifest_path = directory / "manifest.json"
    asins_path = directory / "parent_asins.json"
    embeddings_path = directory / "embeddings.npy"
    if not (manifest_path.is_file() and asins_path.is_file() and embeddings_path.is_file()):
        raise EmbeddingCacheError(
            f"complete embedding cache not found at {directory}; run build_embeddings.py first"
        )

    manifest_raw = _read_json(manifest_path)
    if not isinstance(manifest_raw, dict):
        raise EmbeddingCacheError("embedding cache manifest must be a JSON object")
    manifest = EmbeddingCacheManifest.from_dict(manifest_raw)
    config = resolve_text_config(text_version)
    expected_fingerprint = fingerprint or catalog_text_fingerprint(catalog, config.name)
    expected = {
        "format_version": CACHE_FORMAT_VERSION,
        "model": encoder.model,
        "dimension": encoder.dimension,
        "text_version": config.name,
        "catalog_fingerprint": expected_fingerprint,
        "item_count": len(catalog),
        "normalized": True,
    }
    for field, expected_value in expected.items():
        if getattr(manifest, field) != expected_value:
            raise EmbeddingCacheError(
                f"embedding cache {field} mismatch: "
                f"{getattr(manifest, field)!r} != {expected_value!r}"
            )

    raw_asins = _read_json(asins_path)
    if not isinstance(raw_asins, list) or not all(isinstance(value, str) for value in raw_asins):
        raise EmbeddingCacheError("parent_asins.json must contain a list of strings")
    parent_asins = tuple(raw_asins)
    expected_asins = tuple(catalog)
    if parent_asins != expected_asins:
        raise EmbeddingCacheError("embedding cache ASIN order does not match the catalog")

    try:
        embeddings = np.load(embeddings_path, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise EmbeddingCacheError("cannot load embeddings.npy") from error
    if embeddings.shape != (len(catalog), encoder.dimension):
        raise EmbeddingCacheError(
            f"embedding cache matrix has unexpected shape {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise EmbeddingCacheError("embedding cache matrix must use float32")
    return LoadedEmbeddingCache(
        cache_dir=directory,
        manifest=manifest,
        parent_asins=parent_asins,
        embeddings=embeddings,
        cache_hit=True,
    )


def build_embedding_cache(
    catalog: Catalog,
    encoder: EmbeddingEncoder,
    cache_dir: str | Path,
    *,
    text_version: str = DEFAULT_TEXT_VERSION,
    batch_size: int | None = None,
    workers: int = 1,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> LoadedEmbeddingCache:
    """Batch encode the catalog with resumable progress and a memory-mapped cache."""
    config = resolve_text_config(text_version)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    actual_batch_size = _positive_int(
        batch_size or getattr(encoder, "batch_size", 10),
        "batch_size",
    )
    actual_workers = _positive_int(workers, "workers")
    fingerprint = catalog_text_fingerprint(catalog, config.name)

    if not force:
        try:
            return load_embedding_cache(
                catalog,
                encoder,
                directory,
                text_version=config.name,
                fingerprint=fingerprint,
            )
        except EmbeddingCacheError as error:
            if (directory / "manifest.json").exists():
                raise EmbeddingCacheError(f"{error}; pass force=True to rebuild") from error

    partial_path = directory / "embeddings.partial.npy"
    progress_path = directory / "progress.json"
    progress_raw: dict[str, Any] = {}
    if progress_path.is_file() and partial_path.is_file() and not force:
        value = _read_json(progress_path)
        if isinstance(value, dict):
            progress_raw = value
    resumable = (
        progress_raw.get("model") == encoder.model
        and progress_raw.get("dimension") == encoder.dimension
        and progress_raw.get("text_version") == config.name
        and progress_raw.get("catalog_fingerprint") == fingerprint
        and progress_raw.get("item_count") == len(catalog)
    )
    if resumable:
        try:
            start = int(progress_raw.get("next_index", 0))
        except (TypeError, ValueError) as error:
            raise EmbeddingCacheError("partial embedding progress is invalid") from error
        if not 0 <= start <= len(catalog):
            raise EmbeddingCacheError("partial embedding progress is outside catalog bounds")
        matrix = np.lib.format.open_memmap(partial_path, mode="r+")
        if matrix.shape != (len(catalog), encoder.dimension) or matrix.dtype != np.float32:
            raise EmbeddingCacheError("partial embedding cache has an incompatible shape or dtype")
    else:
        start = 0
        matrix = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(catalog), encoder.dimension),
        )

    products = catalog.items_in_order
    if progress is not None:
        progress(start, len(products))
    offsets = range(start, len(products), actual_batch_size)

    def encode_batch(offset: int) -> tuple[int, int, np.ndarray]:
        batch = products[offset : offset + actual_batch_size]
        texts = [build_product_text(product, config) for product in batch]
        encoded = np.asarray(encoder.encode(texts), dtype=np.float32)
        _validate_embedding_batch(encoded, len(batch), encoder.dimension)
        return offset, len(batch), encoded

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        encoded_batches = executor.map(encode_batch, offsets)
        for offset, batch_length, encoded in encoded_batches:
            matrix[offset : offset + batch_length] = normalize_embeddings(encoded)
            matrix.flush()
            completed = offset + batch_length
            _atomic_json(
                progress_path,
                {
                    "model": encoder.model,
                    "dimension": encoder.dimension,
                    "text_version": config.name,
                    "catalog_fingerprint": fingerprint,
                    "item_count": len(catalog),
                    "next_index": completed,
                },
            )
            if progress is not None:
                progress(completed, len(products))

    del matrix
    embeddings_path = directory / "embeddings.npy"
    partial_path.replace(embeddings_path)
    parent_asins = list(catalog)
    _atomic_json(directory / "parent_asins.json", parent_asins)
    manifest = EmbeddingCacheManifest(
        format_version=CACHE_FORMAT_VERSION,
        model=encoder.model,
        dimension=encoder.dimension,
        text_version=config.name,
        catalog_fingerprint=fingerprint,
        item_count=len(catalog),
        normalized=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(directory / "manifest.json", asdict(manifest))
    if progress_path.exists():
        progress_path.unlink()
    loaded = load_embedding_cache(
        catalog,
        encoder,
        directory,
        text_version=config.name,
        fingerprint=fingerprint,
    )
    loaded.cache_hit = False
    return loaded


__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_CACHE_ROOT",
    "EmbeddingEncoder",
    "EmbeddingCacheError",
    "EmbeddingCacheManifest",
    "LoadedEmbeddingCache",
    "OpenAIEmbeddingConfig",
    "OpenAIEmbeddingEncoder",
    "normalize_embeddings",
    "catalog_text_fingerprint",
    "default_embedding_cache_dir",
    "load_embedding_cache",
    "build_embedding_cache",
]
