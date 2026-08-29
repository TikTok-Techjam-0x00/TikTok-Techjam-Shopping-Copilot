from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace

from src.item import Candidate, Item, RankedCandidate
from src.reranking_plugins import QwenReranker
from src.state import create_state


class _Completions:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=17, completion_tokens=5),
        )


def _client(content: str | None = None, error: Exception | None = None):
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(content, error)))


def _candidates(count: int = 4) -> list[Candidate]:
    return [
        Candidate(
            item=Item(parent_asin=f"B{i:03d}", title=f"Product {i}"),
            bm25_score=10.0 - i,
            dense_score=0.5,
            retrieval_score=1.0 - i / 10,
            retrieval_rank=i + 1,
        )
        for i in range(count)
    ]


class QwenRerankerTest(unittest.TestCase):
    def test_filters_deduplicates_and_fills_from_retrieval(self) -> None:
        candidates = _candidates()
        state = create_state("session")
        before_state = state.to_dict()
        before_candidates = copy.deepcopy([candidate.to_dict() for candidate in candidates])
        content = json.dumps(["B002", "B999", "B002"])

        ranked = QwenReranker(client=_client(content)).rerank(state, candidates, top_k=3)

        self.assertEqual([item.parent_asin for item in ranked], ["B002", "B000", "B001"])
        self.assertTrue(all(isinstance(item, RankedCandidate) for item in ranked))
        self.assertEqual([item.rerank_rank for item in ranked], [1, 2, 3])
        self.assertLessEqual(len(ranked), 3)
        self.assertTrue({item.parent_asin for item in ranked} <= {item.parent_asin for item in candidates})
        self.assertEqual(state.to_dict(), before_state)
        self.assertEqual([candidate.to_dict() for candidate in candidates], before_candidates)

    def test_api_failure_returns_retrieval_top_k(self) -> None:
        candidates = _candidates()
        reranker = QwenReranker(client=_client(error=TimeoutError("timeout")))

        ranked = reranker.rerank(create_state("session"), candidates, top_k=2)

        self.assertEqual([item.parent_asin for item in ranked], ["B000", "B001"])
        self.assertEqual(reranker.last_prompt_tokens, 0)
        self.assertEqual(reranker.last_completion_tokens, 0)

    def test_usage_is_recorded_on_success(self) -> None:
        reranker = QwenReranker(client=_client(json.dumps(["B001", "B000"])))
        reranker.rerank(create_state("session"), _candidates(), top_k=2)
        self.assertEqual(reranker.last_prompt_tokens, 17)
        self.assertEqual(reranker.last_completion_tokens, 5)

    def test_prompt_uses_complete_conversation_without_shopping_state(self) -> None:
        client = _client(json.dumps(["B000"]))
        reranker = QwenReranker(client=client)
        conversation = [
            {"role": "user", "content": "I need running shoes."},
            {"role": "assistant", "content": "What color do you prefer?"},
            {"role": "user", "content": "Actually, make them black walking shoes."},
        ]
        reranker.set_conversation(conversation)

        reranker.rerank(create_state("session"), _candidates(), top_k=1)

        request = client.chat.completions.last_kwargs
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["conversation"], conversation)
        self.assertNotIn("shopping_state", payload)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["conversation_order"], "oldest_to_newest")
        self.assertEqual(payload["metric_priority"], ["Hit@10", "reciprocal_rank"])
        system_prompt = request["messages"][0]["content"]
        self.assertIn("USER messages are the only authority", system_prompt)
        self.assertIn("immediately preceding AGENT question", system_prompt)
        self.assertIn("newest explicit correction or override", system_prompt)
        self.assertIn("retrieval_rank and retrieval_score only as final tie-breakers", system_prompt)


if __name__ == "__main__":
    unittest.main()
