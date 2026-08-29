from __future__ import annotations

import unittest

from src.item import Candidate, Item
from src.reranking.experiments import InitialSimpleReranker


class InitialSimpleRerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "intent": "buying",
            "hard_constraint": {},
            "soft_constraint": {},
            "rejected_values": {},
            "no_prefernce": [],
            "user_profile": {},
        }
        self.candidates = [
            Candidate(
                item=Item(parent_asin="HIGH", title="Running shoe"),
                retrieval_score=10.0,
                retrieval_rank=1,
            ),
            Candidate(
                item=Item(parent_asin="LOW", title="Casual shoe"),
                retrieval_score=0.0,
                retrieval_rank=2,
            ),
        ]

    def test_preserves_original_buying_retrieval_weight(self) -> None:
        ranked = InitialSimpleReranker().rank_all(self.state, self.candidates)
        self.assertEqual([candidate.parent_asin for candidate in ranked], ["HIGH", "LOW"])
        self.assertEqual(ranked[0].rerank_score, 0.55)
        self.assertEqual(ranked[1].rerank_score, 0.0)

    def test_rank_all_is_diagnostic_only(self) -> None:
        candidates = [
            Candidate(
                item=Item(parent_asin=f"P{index:02d}", title="shoe"),
                retrieval_score=float(20 - index),
                retrieval_rank=index,
            )
            for index in range(1, 13)
        ]
        reranker = InitialSimpleReranker()
        self.assertEqual(len(reranker.rerank(self.state, candidates)), 10)
        self.assertEqual(len(reranker.rank_all(self.state, candidates)), 12)

    def test_records_historical_source_commits(self) -> None:
        reranker = InitialSimpleReranker()
        self.assertTrue(reranker.historical_source_commit.startswith("799e8c1"))
        self.assertTrue(reranker.composition_adapter_commit.startswith("66c0579"))


if __name__ == "__main__":
    unittest.main()
