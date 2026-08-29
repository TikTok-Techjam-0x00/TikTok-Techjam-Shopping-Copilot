from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attribute import AttributeName, normalize_attribute_map
from src.item import Candidate
from src.retrieval import (
    BM25Retriever,
    Catalog,
    DenseRetriever,
    HybridConfig,
    HybridRetriever,
    MultiVectorConfig,
    MultiVectorDenseRetriever,
    Retriever,
    TEXT_CONFIGS,
    build_bm25_fields,
    build_embedding_cache,
    build_product_text,
    build_retrieval_query,
    candidate_union,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


PRODUCTS = [
    {
        "parent_asin": "RUN-1",
        "title": "Lightweight Road Running Shoes",
        "features": ["breathable mesh"],
        "categories": ["Shoes", "Running"],
        "price": 79.0,
    },
    {
        "parent_asin": "HIKE-1",
        "title": "Waterproof Hiking Boots",
        "features": ["ankle support", "trail grip"],
        "categories": ["Shoes", "Hiking Boots"],
        "price": 99.0,
    },
    {
        "parent_asin": "HIKE-2",
        "title": "Casual Hiking Shoes",
        "features": ["light trail use"],
        "categories": ["Shoes", "Hiking"],
        "price": None,
    },
]


class FakeEmbeddingEncoder:
    model = "fake-semantic-v1"
    dimension = 3
    batch_size = 2

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float("running" in lowered or "road" in lowered),
                    float("hiking" in lowered or "trail" in lowered),
                    float("other" in lowered or "casual" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class InterruptingEmbeddingEncoder(FakeEmbeddingEncoder):
    def encode(self, texts: list[str]) -> np.ndarray:
        if self.calls >= 1:
            raise RuntimeError("simulated provider interruption")
        return super().encode(texts)


class QueryAwareEmbeddingEncoder(FakeEmbeddingEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.query_calls: list[tuple[list[str], str | None]] = []

    def encode_queries(
        self,
        texts: list[str],
        *,
        instruct: str | None = None,
    ) -> np.ndarray:
        self.query_calls.append((list(texts), instruct))
        return super().encode(texts)


class CatalogTest(unittest.TestCase):
    def test_load_normalizes_skips_malformed_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [
                json.dumps(PRODUCTS[0]),
                json.dumps({**PRODUCTS[0], "title": "duplicate"}),
                "not-json",
                json.dumps({"title": "missing identifier"}),
                json.dumps({"parent_asin": "MINIMAL", "features": None}),
            ]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            catalog = Catalog.load(path)

        self.assertEqual(list(catalog), ["RUN-1", "MINIMAL"])
        self.assertEqual(catalog["RUN-1"].title, PRODUCTS[0]["title"])
        self.assertEqual(catalog["MINIMAL"].features, [])
        self.assertEqual(catalog.stats.rows_seen, 5)
        self.assertEqual(catalog.stats.items_loaded, 2)
        self.assertEqual(catalog.stats.duplicate_asins, 1)
        self.assertEqual(catalog.stats.malformed_rows, 2)

    def test_gzip_catalog_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(PRODUCTS[0]) + "\n")
            catalog = Catalog.load(path)
        self.assertEqual(catalog["RUN-1"].parent_asin, "RUN-1")


class QueryConstructionTest(unittest.TestCase):
    def test_state_values_are_authoritative_for_intent_override(self) -> None:
        state = {
            "hard_constraint": {"category": "hiking boots"},
            "soft_constraint": {"feature": ["waterproof", "ankle support"]},
        }
        query = build_retrieval_query(
            "Actually forget running shoes; I need hiking boots.",
            state,
            "buying",
        )
        self.assertEqual(query, "hiking boots waterproof ankle support")
        self.assertNotIn("running", query)

    def test_no_preference_and_numeric_ranges_do_not_pollute_lexical_query(self) -> None:
        state = {
            "hard_constraint": normalize_attribute_map({
                "category": "running shoes",
                "budget": {"max": 100, "unit": "USD"},
            }),
            "soft_constraint": normalize_attribute_map(
                {"brand": "Example", "color": "black"}
            ),
            "no_prefernce": [AttributeName.BRAND],
        }
        self.assertEqual(build_retrieval_query("fallback", state), "running shoes black")

    def test_empty_state_falls_back_to_current_message(self) -> None:
        self.assertEqual(
            build_retrieval_query("  lightweight   shoes ", {"hard_constraint": {}}),
            "lightweight shoes",
        )

    def test_cjk_content_is_removed_at_the_retrieval_boundary(self) -> None:
        state = {
            "hard_constraint": {"category": "hiking boots " + "\u767b\u5c71\u978b"},
            "soft_constraint": {"feature": "waterproof " + "\u9632\u6c34"},
        }
        self.assertEqual(
            build_retrieval_query("ignored", state),
            "hiking boots waterproof",
        )
        self.assertEqual(
            build_retrieval_query("\u5e2e\u6211\u627e running shoes"),
            "running shoes",
        )


class BM25RetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = Catalog.from_items([*PRODUCTS, PRODUCTS[1]])
        self.retriever = BM25Retriever(self.catalog)

    def tearDown(self) -> None:
        self.retriever.close()

    def test_top_k_contract_order_scores_and_deduplication(self) -> None:
        candidates = self.retriever.retrieve("hiking", k=2)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(isinstance(value, Candidate) for value in candidates))
        self.assertEqual(len({value.parent_asin for value in candidates}), 2)
        self.assertEqual([value.retrieval_rank for value in candidates], [1, 2])
        self.assertGreaterEqual(candidates[0].retrieval_score, candidates[1].retrieval_score)
        self.assertEqual(candidates[0].bm25_score, candidates[0].retrieval_score)

    def test_state_drives_retrieval_after_override(self) -> None:
        state = {"hard_constraint": {"category": "hiking boots"}}
        candidates = self.retriever.retrieve("running shoes", state, "buying", k=1)
        self.assertEqual(candidates[0].parent_asin, "HIKE-1")

    def test_empty_query_and_non_positive_k_return_empty(self) -> None:
        self.assertEqual(self.retriever.retrieve("", k=10), [])
        self.assertEqual(self.retriever.retrieve("running", k=0), [])

    def test_output_is_deterministic(self) -> None:
        first = [value.parent_asin for value in self.retriever.retrieve("hiking", k=10)]
        second = [value.parent_asin for value in self.retriever.retrieve("hiking", k=10)]
        self.assertEqual(first, second)

    def test_facade_preserves_configurable_k(self) -> None:
        facade = Retriever(self.retriever)
        self.assertEqual(len(facade.retrieve("hiking", k=1)), 1)

    def test_text_version_excludes_unselected_fields_from_matching(self) -> None:
        title_only = BM25Retriever(self.catalog, text_version="title_v0")
        core = BM25Retriever(self.catalog, text_version="core_v2")
        try:
            self.assertEqual(title_only.retrieve("breathable", k=10), [])
            self.assertEqual(core.retrieve("breathable", k=1)[0].parent_asin, "RUN-1")
        finally:
            title_only.close()
            core.close()


class ProductTextConstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.item = Catalog.from_items(
            [
                {
                    "parent_asin": "TEXT-1",
                    "title": "Black Cotton Running Shirt",
                    "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts"],
                    "features": ["Lightweight"],
                    "details": {"Fabric Type": "Cotton", "Color": "Black", "Brand": "Acme"},
                    "store": "Acme Store",
                    "description": ["Long description"],
                }
            ]
        )["TEXT-1"]

    def test_registered_versions_are_ordered_ablations(self) -> None:
        self.assertEqual(
            list(TEXT_CONFIGS),
            [
                "title_v0",
                "title_category_v1",
                "core_v2",
                "core_attributes_v3",
                "all_fields_v4",
                "dense_attributes_v2",
                "dense_attributes_v2_unlabeled",
                "dense_identity_v1",
                "dense_needs_v1",
            ],
        )
        self.assertEqual(TEXT_CONFIGS["title_v0"].fields, ("title",))
        self.assertIn("attributes", TEXT_CONFIGS["core_attributes_v3"].fields)

    def test_bm25_fields_leave_excluded_columns_empty(self) -> None:
        fields = build_bm25_fields(self.item, "title_category_v1")
        self.assertEqual(fields["title"], "Black Cotton Running Shirt")
        self.assertIn("Shirts", fields["categories"])
        self.assertEqual(fields["features"], "")
        self.assertEqual(fields["details"], "")

    def test_dense_text_is_labeled_and_versioned(self) -> None:
        text = build_product_text(self.item, "core_attributes_v3")
        self.assertIn("Title: Black Cotton Running Shirt", text)
        self.assertIn("Category:", text)
        self.assertIn("Features: Lightweight", text)
        self.assertIn("material Cotton", text)
        self.assertIn("brand Acme", text)
        self.assertNotIn("Description:", text)

    def test_dense_attribute_text_uses_official_fields_and_label_ablation(self) -> None:
        labeled = build_product_text(self.item, "dense_attributes_v2")
        unlabeled = build_product_text(self.item, "dense_attributes_v2_unlabeled")
        self.assertIn("Title: Black Cotton Running Shirt", labeled)
        self.assertIn("Product type: Men Shirts", labeled)
        self.assertIn("Material: Cotton", labeled)
        self.assertIn("Color: Black", labeled)
        self.assertIn("Brand: Acme", labeled)
        self.assertNotIn("Budget:", labeled)
        self.assertNotIn("Other:", labeled)
        self.assertNotIn("Title:", unlabeled)
        self.assertIn("Black Cotton Running Shirt", unlabeled)

    def test_identity_and_needs_texts_are_separate(self) -> None:
        identity = build_product_text(self.item, "dense_identity_v1")
        needs = build_product_text(self.item, "dense_needs_v1")
        self.assertIn("Title:", identity)
        self.assertIn("Brand:", identity)
        self.assertNotIn("Material:", identity)
        self.assertIn("Material:", needs)
        self.assertIn("Features:", needs)
        self.assertNotIn("Title:", needs)


class DenseRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.directory.name) / "cache"
        self.catalog = Catalog.from_items(PRODUCTS)
        self.encoder = FakeEmbeddingEncoder()
        self.cache = build_embedding_cache(
            self.catalog,
            self.encoder,
            self.cache_dir,
            text_version="all_fields_v4",
        )
        self.retriever = DenseRetriever(
            self.catalog,
            self.encoder,
            self.cache_dir,
            text_version="all_fields_v4",
        )

    def tearDown(self) -> None:
        self.retriever.close()
        self.cache.close()
        self.directory.cleanup()

    def test_cache_is_normalized_versioned_and_reused(self) -> None:
        self.assertFalse(self.cache.cache_hit)
        self.assertEqual(self.cache.manifest.text_version, "all_fields_v4")
        norms = np.linalg.norm(self.cache.embeddings, axis=1)
        self.assertTrue(np.allclose(norms, np.ones(len(self.catalog))))
        calls = self.encoder.calls
        loaded = build_embedding_cache(
            self.catalog,
            self.encoder,
            self.cache_dir,
            text_version="all_fields_v4",
        )
        self.assertTrue(loaded.cache_hit)
        self.assertEqual(self.encoder.calls, calls)
        loaded.close()

    def test_dense_top_k_scores_ranks_and_query_cache(self) -> None:
        first = self.retriever.retrieve("trail footwear", k=2)
        calls = self.encoder.calls
        second = self.retriever.retrieve("trail footwear", k=2)
        self.assertEqual(first[0].parent_asin, "HIKE-1")
        self.assertEqual([candidate.retrieval_rank for candidate in first], [1, 2])
        self.assertGreaterEqual(first[0].dense_score, first[1].dense_score)
        self.assertEqual(first[0].dense_score, first[0].retrieval_score)
        self.assertEqual(
            [candidate.parent_asin for candidate in first],
            [candidate.parent_asin for candidate in second],
        )
        self.assertEqual(self.encoder.calls, calls)

    def test_empty_query_and_invalid_k(self) -> None:
        self.assertEqual(self.retriever.retrieve("", k=100), [])
        self.assertEqual(self.retriever.retrieve("running", k=0), [])
        with self.assertRaises(TypeError):
            self.retriever.retrieve("running", k=1.5)  # type: ignore[arg-type]

    def test_query_instruction_uses_query_encoder_without_rebuilding_documents(self) -> None:
        query_encoder = QueryAwareEmbeddingEncoder()
        retriever = DenseRetriever(
            self.catalog,
            query_encoder,
            self.cache_dir,
            text_version="all_fields_v4",
            query_embedding_mode="query_instruction",
            query_instruction="Retrieve matching products.",
        )
        try:
            result = retriever.retrieve("trail footwear", k=1)
        finally:
            retriever.close()
        self.assertEqual(result[0].parent_asin, "HIKE-1")
        self.assertEqual(
            query_encoder.query_calls,
            [(["trail footwear"], "Retrieve matching products.")],
        )

    def test_query_mode_omits_instruction(self) -> None:
        query_encoder = QueryAwareEmbeddingEncoder()
        retriever = DenseRetriever(
            self.catalog,
            query_encoder,
            self.cache_dir,
            text_version="all_fields_v4",
            query_embedding_mode="query",
        )
        try:
            retriever.retrieve("running", k=1)
        finally:
            retriever.close()
        self.assertEqual(query_encoder.query_calls, [(["running"], None)])

    def test_partial_cache_resumes_from_last_completed_batch(self) -> None:
        resume_dir = Path(self.directory.name) / "resume-cache"
        interrupted = InterruptingEmbeddingEncoder()
        with self.assertRaisesRegex(RuntimeError, "simulated provider interruption"):
            build_embedding_cache(
                self.catalog,
                interrupted,
                resume_dir,
                text_version="all_fields_v4",
                batch_size=2,
            )

        resumed = FakeEmbeddingEncoder()
        cache = build_embedding_cache(
            self.catalog,
            resumed,
            resume_dir,
            text_version="all_fields_v4",
            batch_size=2,
            workers=2,
        )
        self.assertEqual(resumed.calls, 1)
        self.assertEqual(cache.embeddings.shape, (3, 3))
        cache.close()


class HybridRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        catalog = Catalog.from_items(
            [
                {"parent_asin": "A", "title": "Alpha"},
                {"parent_asin": "B", "title": "Beta"},
                {"parent_asin": "C", "title": "Gamma"},
            ]
        )
        self.bm25 = [
            Candidate(item=catalog["A"], bm25_score=10.0, retrieval_score=10.0, retrieval_rank=1),
            Candidate(item=catalog["B"], bm25_score=5.0, retrieval_score=5.0, retrieval_rank=2),
        ]
        self.dense = [
            Candidate(item=catalog["B"], dense_score=0.9, retrieval_score=0.9, retrieval_rank=1),
            Candidate(item=catalog["C"], dense_score=0.8, retrieval_score=0.8, retrieval_rank=2),
        ]

    def test_candidate_union_deduplicates_and_preserves_raw_scores(self) -> None:
        pool = candidate_union(self.bm25, self.dense)
        self.assertEqual([candidate.parent_asin for candidate in pool], ["A", "B", "C"])
        self.assertEqual(pool[1].bm25_score, 5.0)
        self.assertEqual(pool[1].dense_score, 0.9)
        self.assertIsNone(pool[1].retrieval_score)

    def test_rrf_rewards_products_found_by_both_sources(self) -> None:
        fused = reciprocal_rank_fusion(self.bm25, self.dense, k=3, rank_constant=60)
        self.assertEqual(fused[0].parent_asin, "B")
        self.assertEqual([candidate.retrieval_rank for candidate in fused], [1, 2, 3])
        self.assertGreater(fused[0].retrieval_score, fused[1].retrieval_score)

    def test_weighted_fusion_normalizes_each_source_before_combining(self) -> None:
        lexical = weighted_score_fusion(self.bm25, self.dense, k=3, alpha=1.0)
        semantic = weighted_score_fusion(self.bm25, self.dense, k=3, alpha=0.0)
        self.assertEqual(lexical[0].parent_asin, "A")
        self.assertEqual(semantic[0].parent_asin, "B")
        self.assertTrue(all(0.0 <= candidate.retrieval_score <= 1.0 for candidate in lexical))

    def test_hybrid_retriever_returns_strict_top_k(self) -> None:
        class StaticSource:
            def __init__(self, values: list[Candidate]) -> None:
                self.values = values

            def retrieve(self, query, state=None, intent=None, k=100):
                del query, state, intent
                return self.values[:k]

        hybrid = HybridRetriever(
            StaticSource(self.bm25),
            StaticSource(self.dense),
            config=HybridConfig(method="rrf", source_k=2),
        )
        result = hybrid.retrieve("query", k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].parent_asin, "B")


class MultiVectorDenseRetrieverTest(unittest.TestCase):
    def test_weighted_and_max_fusion_return_ranked_candidates(self) -> None:
        catalog = Catalog.from_items(PRODUCTS)
        encoder = FakeEmbeddingEncoder()
        with tempfile.TemporaryDirectory() as directory:
            identity_dir = Path(directory) / "identity"
            needs_dir = Path(directory) / "needs"
            identity_cache = build_embedding_cache(
                catalog,
                encoder,
                identity_dir,
                text_version="dense_identity_v1",
            )
            identity_cache.close()
            needs_cache = build_embedding_cache(
                catalog,
                encoder,
                needs_dir,
                text_version="dense_needs_v1",
            )
            needs_cache.close()
            for fusion in ("weighted", "max"):
                retriever = MultiVectorDenseRetriever(
                    catalog,
                    encoder,
                    identity_dir,
                    needs_dir,
                    query_embedding_mode="symmetric",
                    config=MultiVectorConfig(fusion=fusion),
                )
                try:
                    result = retriever.retrieve("trail hiking", k=2)
                finally:
                    retriever.close()
                self.assertEqual(result[0].parent_asin, "HIKE-1")
                self.assertEqual([value.retrieval_rank for value in result], [1, 2])


if __name__ == "__main__":
    unittest.main()
