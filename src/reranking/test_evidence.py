from __future__ import annotations

import unittest

from src.item import Candidate, Item
from src.reranking.evidence import EvidenceCoverageReranker


def _candidate(
    parent_asin: str,
    features: list[str],
    *,
    rating_number: int = 10,
) -> Candidate:
    return Candidate(
        item=Item(
            parent_asin=parent_asin,
            title="Example product",
            features=features,
            rating_number=rating_number,
            average_rating=4.5,
        ),
        retrieval_score=1.0,
    )


class EvidenceCoverageRerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reranker = EvidenceCoverageReranker()

    def test_contiguous_phrase_beats_scattered_words(self) -> None:
        state = {"soft_constraint": {"feature": "water resistant expedition"}}
        scattered = _candidate(
            "scattered",
            ["Water-ready shell", "Resistant finish", "Expedition style"],
            rating_number=10_000,
        )
        phrase = _candidate(
            "phrase",
            ["Water resistant expedition shell"],
            rating_number=1,
        )

        ranked = self.reranker.rerank(state, [scattered, phrase])

        self.assertEqual(ranked[0].parent_asin, "phrase")

    def test_exact_catalog_field_breaks_an_otherwise_equal_tie(self) -> None:
        state = {"soft_constraint": {"feature": "pull on closure"}}
        extended = _candidate(
            "extended",
            ["Easy pull on closure style"],
            rating_number=10_000,
        )
        exact = _candidate(
            "exact",
            ["Pull On closure"],
            rating_number=1,
        )

        ranked = self.reranker.rerank(state, [extended, exact])

        self.assertEqual(ranked[0].parent_asin, "exact")

    def test_imported_is_treated_as_product_evidence(self) -> None:
        state = {"soft_constraint": {"feature": "Imported"}}
        domestic = _candidate("domestic", ["Made locally"], rating_number=10_000)
        imported = _candidate("imported", ["Imported"], rating_number=1)

        ranked = self.reranker.rerank(state, [domestic, imported])

        self.assertEqual(ranked[0].parent_asin, "imported")

    def test_single_digit_percentage_is_not_discarded(self) -> None:
        state = {"soft_constraint": {"material": "5% spandex"}}
        wrong_ratio = _candidate(
            "ten-percent",
            ["10% spandex"],
            rating_number=10_000,
        )
        exact_ratio = _candidate(
            "five-percent",
            ["5% spandex"],
            rating_number=1,
        )

        ranked = self.reranker.rerank(state, [wrong_ratio, exact_ratio])

        self.assertEqual(ranked[0].parent_asin, "five-percent")

    def test_single_letter_size_is_preserved_inside_phrase(self) -> None:
        state = {"soft_constraint": {"size": "size M"}}
        wrong_size = _candidate("large", ["size L"], rating_number=10_000)
        exact_size = _candidate("medium", ["size M"], rating_number=1)

        ranked = self.reranker.rerank(state, [wrong_size, exact_size])

        self.assertEqual(ranked[0].parent_asin, "medium")

if __name__ == "__main__":
    unittest.main()
