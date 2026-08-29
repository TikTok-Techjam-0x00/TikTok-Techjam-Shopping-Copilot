"""High-information clarification policy used by the integrated pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..item import Candidate
from .three_b import (
    AskDecision,
    ShoppingStateInput,
    _turn_number,
    decide_ask,
)


_OPEN_REQUIREMENT_MESSAGES = {
    1: "Are there any must-have details I should prioritize?",
    2: "What features or preferences matter most to you?",
    3: "Any final requirements or deal-breakers I should consider?",
}


def decide_high_information_ask(
    shopping_state: ShoppingStateInput,
    candidates_100: Sequence[Candidate | Mapping[str, Any]],
) -> AskDecision:
    """Use three open questions, then return to the standard 3B policy.

    ``other`` can disclose constraints whose catalog field is not known in
    advance. Three turns capture the public-set gain while keeping later
    questions attribute-specific. The final turn is recommendation-only.
    """

    turn = _turn_number(shopping_state)
    if turn <= 3:
        return {
            "ask_attribute": "other",
            "message": _OPEN_REQUIREMENT_MESSAGES[turn],
        }
    if turn >= 10:
        return {"ask_attribute": None, "message": ""}
    return decide_ask(shopping_state, candidates_100)


__all__ = ["decide_high_information_ask"]
