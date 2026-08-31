from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.pipeline.pipeline import Pipeline
from src.retrieval.embedding import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
from src.state.qwen import QwenSemanticResolver
from src.state.semantic import SemanticRequest


class ModelUsageTest(unittest.TestCase):
    def test_qwen_resolver_tracks_and_resets_provider_usage(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=17, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response),
            )
        )
        resolver = QwenSemanticResolver(api_key="test", client=client)
        request = SemanticRequest(
            message="something similar",
            current_state={},
            recent_history=(),
            asked_attribute=None,
            rule_intent="browsing",
            rule_intent_confidence=0.5,
            rule_evidence=(),
            fallback_reasons=("comparison",),
        )

        resolver._call(request)
        self.assertEqual(resolver.model_usage(), (17, 5))
        resolver.reset_usage()
        self.assertEqual(resolver.model_usage(), (0, 0))

    def test_openai_compatible_embedding_tracks_prompt_tokens(self) -> None:
        encoder = OpenAIEmbeddingEncoder(
            OpenAIEmbeddingConfig(
                api_key="test",
                base_url="https://example.invalid/v1",
                dimension=2,
            )
        )
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11),
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])],
        )
        encoder._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **_: response),
        )

        matrix = encoder.encode(["shoe"])

        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(encoder.model_usage(), (11, 0))

    def test_dashscope_query_embedding_tracks_total_tokens(self) -> None:
        encoder = OpenAIEmbeddingEncoder(
            OpenAIEmbeddingConfig(
                api_key="test",
                base_url="https://example.invalid/v1",
                dashscope_base_url="https://example.invalid/api/v1",
                dimension=2,
            )
        )
        body = {
            "usage": {"total_tokens": 13},
            "output": {
                "embeddings": [
                    {"text_index": 0, "embedding": [0.0, 1.0]},
                ]
            },
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")

        with patch("src.retrieval.embedding.urllib.request.urlopen", return_value=response):
            matrix = encoder.encode_queries(["boot"], instruct="retrieve")

        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(encoder.model_usage(), (13, 0))

    def test_pipeline_combines_state_and_embedding_usage(self) -> None:
        state = SimpleNamespace(model_usage=lambda: (20, 4))
        retrieval = SimpleNamespace(model_usage=lambda: (7, 0))

        self.assertEqual(
            Pipeline._combined_model_usage(state, retrieval, None),
            (27, 4),
        )


if __name__ == "__main__":
    unittest.main()
