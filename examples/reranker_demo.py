"""Run module 3A with simulated state and retrieval candidates."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item import Candidate
from src.reranking import recommendations_from_ranking, rerank


@dataclass(frozen=True)
class SimulatedShoppingState:
    """Stand-in used only until module 2's real shopping_state is available."""

    session_id: str
    user_profile: dict[str, Any]
    user_message: str
    turn: int
    intent: str
    hard_constraint: dict[str, Any]
    soft_constraint: dict[str, Any]
    no_prefernce: list[str]


SIMULATED_PROFILE = {
    "preference_tags": ["comfort", "durability"],
    "average_prior_rating": 4.5,
    "purchase_frequency": "3-4 prior purchases",
    "rating_style": "usually positive",
    "summary": "Prior purchases emphasize comfort and durability.",
}

SIMULATED_SHOPPING_STATE = SimulatedShoppingState(
    session_id="demo-session",
    user_profile=SIMULATED_PROFILE,
    user_message="I want black lightweight running shoes under $100.",
    turn=2,
    intent="buying",
    hard_constraint={
        "category": "running shoes",
        "budget": {"max": 100},
        "material": "mesh",
    },
    soft_constraint={
        "color": "black",
        "feature": ["lightweight", "comfortable"],
    },
    no_prefernce=["brand"],
)

SIMULATED_CANDIDATES_100 = [Candidate.from_dict(value) for value in [
    {
        "parent_asin": "B-LEATHER",
        "bm25_score": 9.2,
        "dense_score": 0.82,
        "retrieval_score": 0.95,
        "retrieval_rank": 1,
        "product": {
            "title": "Black Leather Running Shoes",
            "categories": ["Shoes", "Athletic", "Running"],
            "features": ["Comfortable cushioned sole"],
            "details": {"Material": "Leather", "Color": "Black"},
            "description": ["Road running shoes for daily training."],
            "price": 69.99,
            "average_rating": 4.6,
            "rating_number": 850,
            "store": "Example Sports",
        },
    },
    {
        "parent_asin": "A-MESH",
        "bm25_score": 8.7,
        "dense_score": 0.91,
        "retrieval_score": 0.88,
        "retrieval_rank": 2,
        "product": {
            "title": "Black Lightweight Mesh Running Shoes",
            "categories": ["Shoes", "Athletic", "Running"],
            "features": ["Lightweight", "Comfortable breathable upper"],
            "details": {"Material": "Mesh", "Color": "Black"},
            "description": ["Cushioned running shoes for long walks and training."],
            "price": 79.99,
            "average_rating": 4.5,
            "rating_number": 420,
            "store": "Demo Run",
        },
    },
    {
        "parent_asin": "C-WHITE",
        "bm25_score": 6.1,
        "dense_score": 0.65,
        "retrieval_score": 0.82,
        "retrieval_rank": 3,
        "product": {
            "title": "White Canvas Casual Sneakers",
            "categories": ["Shoes", "Fashion Sneakers"],
            "features": ["Everyday casual style"],
            "details": {"Material": "Canvas", "Color": "White"},
            "description": ["A relaxed sneaker for casual wear."],
            "price": 49.99,
            "average_rating": 4.4,
            "rating_number": 210,
            "store": "Demo Casual",
        },
    },
]]


def main() -> None:
    candidates_10 = rerank(
        SIMULATED_SHOPPING_STATE,
        SIMULATED_CANDIDATES_100,
        top_k=10,
    )
    print("Internal candidates_10:")
    print(json.dumps([value.to_dict() for value in candidates_10], indent=2, ensure_ascii=False))
    print("\nOfficial recommendations:")
    print(json.dumps(recommendations_from_ranking(candidates_10), indent=2))


if __name__ == "__main__":
    main()
