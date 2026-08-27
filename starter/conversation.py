from __future__ import annotations

import re

from starter.intent import classify_intent
from starter.slots import extract_slots
from starter.state import ShoppingState


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def sanitize_retrieval_text(value: object) -> str:
    """Remove CJK content before text reaches any retrieval implementation."""

    without_cjk = CJK_RE.sub(" ", str(value or ""))
    return " ".join(without_cjk.split())


def create_state(user_profile: dict | None = None) -> ShoppingState:
    """Create isolated session state with profile data kept as soft context."""

    return ShoppingState(user_profile=dict(user_profile or {}))


def update_state(
    previous_state: ShoppingState,
    user_message: str,
    *,
    turn: int | None = None,
    asked_attribute: str | None = None,
) -> ShoppingState:
    """Update a session in place and return it for convenient pipeline chaining."""

    previous_intent = previous_state.intent if previous_state.history else "unknown"
    intent = classify_intent(user_message, previous_intent=previous_intent)
    update = extract_slots(
        user_message,
        asked_attribute=asked_attribute,
        is_override=intent.is_override,
    )
    if intent.intent != "unknown":
        update.intent = intent.intent
    previous_state.apply_update(update)
    previous_state.intent_confidence = intent.confidence
    previous_state.turn_count = turn if turn is not None else previous_state.turn_count + 1
    previous_state.history.append(user_message)
    return previous_state


def retrieval_query(state: ShoppingState) -> str:
    """Distill current state into stable text for keyword/dense retrieval."""

    parts: list[str] = []
    if state.category:
        parts.append(state.category)
    for attribute, value in state.hard_constraints.items():
        if attribute == "budget":
            continue
        parts.append(str(value))
    parts.extend(state.soft_preferences)
    return sanitize_retrieval_text(
        " ".join(dict.fromkeys(part for part in parts if part))
    )
