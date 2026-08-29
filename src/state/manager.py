"""Conversation-state lifecycle and retrieval-safe context distillation."""

from __future__ import annotations

import re

from ..retrieval import build_retrieval_query
from .intent import classify_intent
from .model import ShoppingState
from .semantic import (
    SemanticPolicy,
    SemanticResolver,
    build_semantic_request,
    decide_semantic_fallback,
    merge_rule_and_semantic,
    resolve_final_intent,
    validate_semantic_resolution,
)
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
    semantic_resolver: SemanticResolver | None = None,
    semantic_policy: SemanticPolicy | None = None,
) -> ShoppingState:
    previous_intent = state.intent if state.history else "unknown"
    result = classify_intent(user_message, previous_intent)
    rule_update = extract_slots(
        user_message,
        asked_attribute=asked_attribute,
        is_override=result.is_override,
    )
    if result.intent != "unknown":
        rule_update.intent = result.intent

    selected_policy = semantic_policy or SemanticPolicy()
    decision = decide_semantic_fallback(
        user_message,
        state,
        result,
        rule_update,
        asked_attribute=asked_attribute,
        policy=selected_policy,
    )
    state.semantic_fallback_used = False
    state.semantic_fallback_reasons = list(decision.reasons)
    state.semantic_validation_errors = []
    update = rule_update
    semantic_resolution = None
    if decision.should_resolve and semantic_resolver is not None:
        state.semantic_fallback_used = True
        state.semantic_fallback_count += 1
        request = build_semantic_request(
            user_message,
            state,
            result,
            decision,
            asked_attribute=asked_attribute,
            policy=selected_policy,
        )
        try:
            raw_resolution = semantic_resolver.resolve(request)
        except Exception:
            raw_resolution = None
            state.semantic_validation_errors = ["semantic_resolver_failed"]
        if raw_resolution is not None:
            semantic_resolution, validation_errors = validate_semantic_resolution(
                raw_resolution,
                policy=selected_policy,
            )
            state.semantic_validation_errors = list(validation_errors)
        elif not state.semantic_validation_errors:
            state.semantic_validation_errors = ["empty_or_invalid_semantic_result"]
        if semantic_resolution is not None:
            update = merge_rule_and_semantic(
                rule_update,
                semantic_resolution.update,
                prefer_semantic_intent=(
                    result.intent == "unknown"
                    or result.confidence < selected_policy.intent_confidence_threshold
                ),
            )

    final_intent = resolve_final_intent(
        result,
        semantic_resolution,
        state,
        policy=selected_policy,
    )
    update.intent = final_intent.intent
    next_intent = final_intent.intent
    intent_changed = bool(state.history) and next_intent != state.intent
    effective_override = update.override or intent_changed
    update.override = effective_override
    update.clear_soft_constraint = update.clear_soft_constraint or effective_override
    if intent_changed and next_intent == "browsing":
        update.clear_hard_constraint = True
    state.apply_update(update)
    state.intent_confidence = final_intent.confidence
    state.intent_resolution_source = final_intent.source
    state.intent_smoothed = final_intent.smoothed
    state.user_message = user_message
    state.turn = turn if turn is not None else state.turn + 1
    if intent_changed:
        state.intent_transitions.append({
            "turn": state.turn,
            "from": previous_intent,
            "to": next_intent,
        })
    state.history.append(user_message)
    return state


def retrieval_query(state: ShoppingState) -> str:
    return sanitize_retrieval_text(build_retrieval_query("", state, state.intent))
