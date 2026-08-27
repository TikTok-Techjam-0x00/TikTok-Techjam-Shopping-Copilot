"""Conversation-state lifecycle and retrieval-safe context distillation."""

from __future__ import annotations

import re

from ..retrieval import build_retrieval_query
from .intent import classify_intent
from .model import ShoppingState
from .slots import extract_slots


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def sanitize_retrieval_text(value: object) -> str:
    """Remove CJK content before text reaches keyword or dense retrieval."""

    return " ".join(CJK_RE.sub(" ", str(value or "")).split())


def create_state(session_id: str, user_profile: dict | None = None) -> ShoppingState:
    return ShoppingState(session_id=session_id, user_profile=dict(user_profile or {}))


def update_state(
    state: ShoppingState,
    user_message: str,
    *,
    turn: int | None = None,
    asked_attribute: str | None = None,
) -> ShoppingState:
    previous_intent = state.intent if state.history else "unknown"
    result = classify_intent(user_message, previous_intent)
    update = extract_slots(user_message, asked_attribute=asked_attribute, is_override=result.is_override)
    if result.intent != "unknown":
        update.intent = result.intent
    state.apply_update(update)
    state.intent_confidence = result.confidence
    state.user_message = user_message
    state.turn = turn if turn is not None else state.turn + 1
    state.history.append(user_message)
    return state


def retrieval_query(state: ShoppingState) -> str:
    return sanitize_retrieval_text(build_retrieval_query("", state, state.intent))
