"""Module 3A: candidate reranking and evaluator-output adapters."""

from ..item import (
    Candidate,
    Candidates10,
    Candidates100,
    Item,
    RankedCandidate,
    candidate,
    candidates_10,
    candidates_100,
    item,
    reranked_candidate,
)
from .constraint_matcher import (
    CandidateConstraintMatches,
    ConstraintMatch,
    ConstraintMatcher,
    ConstraintMatcherConfig,
    MatchStatus,
    MultiValuePolicy,
    match_constraint,
)
from .feature_extractor import (
    CandidateFeatureExtractor,
    CandidateSignals,
    ConstraintFeatureWeights,
    extract_candidate_features,
)
from .scorers import (
    RelevanceScore,
    RelevanceScorer,
    RuleFuzzyScorer,
    RuleFuzzyScorerConfig,
    score_rule_relevance,
)
from .reranker import (
    HardConstraintStrategy,
    RerankerStrategyConfig,
    SimpleReranker,
    recommendations_from_ranking,
    rerank,
)

__all__ = [
    "Item",
    "Candidate",
    "RankedCandidate",
    "Candidates100",
    "Candidates10",
    "item",
    "candidate",
    "reranked_candidate",
    "candidates_100",
    "candidates_10",
    "MatchStatus",
    "MultiValuePolicy",
    "ConstraintMatch",
    "CandidateConstraintMatches",
    "ConstraintMatcherConfig",
    "ConstraintMatcher",
    "match_constraint",
    "ConstraintFeatureWeights",
    "CandidateSignals",
    "CandidateFeatureExtractor",
    "extract_candidate_features",
    "RelevanceScore",
    "RelevanceScorer",
    "RuleFuzzyScorerConfig",
    "RuleFuzzyScorer",
    "score_rule_relevance",
    "HardConstraintStrategy",
    "RerankerStrategyConfig",
    "SimpleReranker",
    "rerank",
    "recommendations_from_ranking",
]
