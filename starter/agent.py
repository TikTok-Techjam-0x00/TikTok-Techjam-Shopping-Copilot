"""Official Agent adapter for the shared team pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import Pipeline


class Agent:
    """Official evaluator entry point for the shopping copilot."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.pipeline = Pipeline(catalog_path)

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
