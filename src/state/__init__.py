"""Module 2: intent routing, slot extraction, and conversation state."""

from .intent import Intent, IntentResult, classify_intent
from .manager import create_state, retrieval_query, sanitize_retrieval_text, update_state
from .model import ShoppingState, StateUpdate
from .semantic import (
    CallableSemanticResolver,
    SemanticDecision,
    SemanticPolicy,
    SemanticRequest,
    SemanticResolution,
    SemanticResolver,
    decide_semantic_fallback,
)
from .slots import extract_slots

__all__ = [
    "Intent",
    "IntentResult",
    "ShoppingState",
    "StateUpdate",
    "CallableSemanticResolver",
    "SemanticDecision",
    "SemanticPolicy",
    "SemanticRequest",
    "SemanticResolution",
    "SemanticResolver",
    "classify_intent",
    "create_state",
    "decide_semantic_fallback",
    "extract_slots",
    "retrieval_query",
    "sanitize_retrieval_text",
    "update_state",
]
