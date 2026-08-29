"""Pluggable relevance scorers for Module 3A."""

from .base import RelevanceScore, RelevanceScorer
from .rule_scorer import (
    FastRuleFuzzyScorer,
    RuleFuzzyScorer,
    RuleFuzzyScorerConfig,
    score_rule_relevance,
)

__all__ = [
    "RelevanceScore",
    "RelevanceScorer",
    "FastRuleFuzzyScorer",
    "RuleFuzzyScorerConfig",
    "RuleFuzzyScorer",
    "score_rule_relevance",
]
