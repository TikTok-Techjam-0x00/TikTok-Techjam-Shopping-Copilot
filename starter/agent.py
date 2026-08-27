from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from starter.conversation import (
    create_state,
    retrieval_query,
    sanitize_retrieval_text,
    update_state,
)
from starter.state import ShoppingState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Stateful shopping agent with a deterministic BM25 fallback."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, ShoppingState] = {}
        self._last_asked: dict[str, str | None] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Each evaluator session owns independent mutable state.
        self._sessions[session_id] = create_state(user_profile)
        self._last_asked[session_id] = None

    def get_state(self, session_id: str) -> dict:
        """Expose a serializable state contract for retrieval/dialogue modules."""

        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id].to_dict()

    @staticmethod
    def _next_attribute(state: ShoppingState, turn: int) -> str | None:
        """Minimal fallback policy until the team's dialogue module is wired in."""

        if turn >= 10:
            return None
        profile_tags = {
            str(tag).lower()
            for tag in state.user_profile.get("preference_tags", [])
        }
        profile_order = []
        for tags, attribute in (
            ({"fit"}, "size"),
            ({"material"}, "material"),
            ({"weather"}, "use_case"),
            ({"style"}, "style"),
            ({"comfort", "durability", "performance", "warmth"}, "feature"),
        ):
            if profile_tags & tags:
                profile_order.append(attribute)
        order = [*profile_order, "feature", "material", "size", "use_case", "color", "brand", "budget", "style", "other"]
        known = set(state.known_attributes)
        excluded = known | state.asked_attributes | state.rejected_attributes
        return next((attribute for attribute in dict.fromkeys(order) if attribute not in excluded), None)

    @staticmethod
    def _question(attribute: str | None) -> str:
        prompts = {
            "feature": "Which product feature matters most to you?",
            "material": "Do you have a material preference?",
            "size": "What size or fit do you need?",
            "use_case": "What will you mainly use it for?",
            "color": "Do you have a color preference?",
            "brand": "Do you prefer a particular brand?",
            "budget": "What is your maximum budget?",
            "style": "What style do you prefer?",
            "other": "What other requirement matters most to you?",
        }
        return prompts.get(attribute, "Here are the closest matches I found.")

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = update_state(
            self._sessions[session_id],
            user_message,
            turn=turn,
            asked_attribute=self._last_asked.get(session_id),
        )
        query = retrieval_query(state) or sanitize_retrieval_text(user_message)
        unique_terms = list(dict.fromkeys(_terms(query)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        ask_attribute = self._next_attribute(state, turn)
        if ask_attribute:
            state.mark_attribute_asked(ask_attribute)
        self._last_asked[session_id] = ask_attribute
        return {
            "message": self._question(ask_attribute),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
