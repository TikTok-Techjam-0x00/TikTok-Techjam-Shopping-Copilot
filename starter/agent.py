"""Official Agent adapter for the shared team pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import Pipeline
from src.retrieval import Catalog
from src.state import (
    SemanticPolicy,
    SemanticResolver,
    qwen_semantic_resolver_from_env,
)


_AUTO_RESOLVER = object()


class Agent:
    """Official evaluator entry point for the shopping copilot."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        catalog: Catalog | None = None,
        semantic_resolver: SemanticResolver | None | object = _AUTO_RESOLVER,
        semantic_policy: SemanticPolicy | None = None,
    ) -> None:
        configured_resolver = (
            qwen_semantic_resolver_from_env()
            if semantic_resolver is _AUTO_RESOLVER
            else semantic_resolver
        )
        self.pipeline = Pipeline(
            catalog_path,
            catalog=catalog,
            semantic_resolver=configured_resolver,
            semantic_policy=semantic_policy,
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.pipeline.reset(session_id, user_profile)

    def get_state(self, session_id: str) -> dict:
        return self.pipeline.get_state(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return self.pipeline.respond(session_id, user_message, turn, top_k)
