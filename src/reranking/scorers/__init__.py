"""Pluggable relevance scorers for Module 3A."""

from .base import RelevanceScore, RelevanceScorer
from .rule_scorer import (
    RuleFuzzyScorer,
    RuleFuzzyScorerConfig,
    score_rule_relevance,
)

__all__ = [
    "RelevanceScore",
    "RelevanceScorer",
    "RuleFuzzyScorerConfig",
    "RuleFuzzyScorer",
    "score_rule_relevance",
]
