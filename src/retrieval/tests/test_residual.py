from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from src.item import Candidate, Item
from src.retrieval.residual import LexicalGatedResidualRetriever, ResidualDenseConfig
from src.retrieval.retriever import Retriever


def _candidate(index: int, *, dense: bool = False) -> Candidate:
    return Candidate(
        item=Item(parent_asin=f"P{index:04d}", title=f"Product {index}"),
        bm25_score=None if dense else float(2000 - index),
        dense_score=float(2000 - index) if dense else None,
        retrieval_score=float(2000 - index),
        retrieval_rank=index,
    )


class _StaticRetriever:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def retrieve(self, query, state=None, intent=None, k=100):
        del query, state, intent
        return self.candidates[:k]


class ResidualDenseRetrieverTest(unittest.TestCase):
    def test_dense_only_candidates_cannot_enter(self) -> None:
        lexical = [_candidate(index) for index in range(1, 401)]
        semantic = [_candidate(9999, dense=True), _candidate(350, dense=True)]
        retriever = LexicalGatedResidualRetriever(
            _StaticRetriever(lexical),
            _StaticRetriever(semantic),
            config=ResidualDenseConfig(
                protected_lexical=80,
                semantic_slots=20,
                semantic_source_k=100,
                lexical_gate_depth=400,
                minimum_lexical_rank=281,
            ),
        )

        result = retriever.retrieve_residual_page("query", page=2, page_size=100)

        self.assertEqual([value.parent_asin for value in result[:80]], [
            value.parent_asin for value in lexical[200:280]
        ])
        self.assertIn("P0350", [value.parent_asin for value in result])
        self.assertNotIn("P9999", [value.parent_asin for value in result])

    def test_ordinary_retrieve_is_exact_lexical_delegate(self) -> None:
        lexical = [_candidate(index) for index in range(1, 151)]
        retriever = LexicalGatedResidualRetriever(
            _StaticRetriever(lexical),
            _StaticRetriever([]),
        )
        self.assertEqual(retriever.retrieve("query", k=100), lexical[:100])

    def test_focused_cohort_does_not_backfill_the_page(self) -> None:
        lexical = [_candidate(index) for index in range(1, 401)]
        semantic = [_candidate(350, dense=True)]
        retriever = LexicalGatedResidualRetriever(
            _StaticRetriever(lexical),
            _StaticRetriever(semantic),
            config=ResidualDenseConfig(
                protected_lexical=10,
                semantic_slots=10,
                semantic_source_k=100,
                lexical_gate_depth=400,
                minimum_lexical_rank=281,
                fill_lexical_tail=False,
            ),
        )

        result = retriever.retrieve_residual_page("query", page=2, page_size=100)

        self.assertEqual(len(result), 11)
        self.assertEqual(result[-1].parent_asin, "P0350")

    def test_sota_factory_falls_back_when_needs_cache_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                '{"parent_asin":"P1","title":"Trail shoe"}\n',
                encoding="utf-8",
            )
            retriever = Retriever.sota_semantic_residual(
                str(catalog),
                cache_root=root / "missing",
            )

            self.assertEqual(
                retriever.retrieve("trail shoe", k=1)[0].parent_asin,
                "P1",
            )


if __name__ == "__main__":
    unittest.main()
