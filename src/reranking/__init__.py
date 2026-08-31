"""Module 3A: candidate reranking and evaluator-output adapters."""

from ..item import (
    Candidate,
    Candidates10,
    Candidates100,
    Item,
    RankedCandidate,
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
    FastRuleFuzzyScorer,
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
from .evidence import EvidenceCoverageReranker

__all__ = [
    "Item",
    "Candidate",
    "RankedCandidate",
    "Candidates100",
    "Candidates10",
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
    "FastRuleFuzzyScorer",
    "RuleFuzzyScorerConfig",
    "RuleFuzzyScorer",
    "score_rule_relevance",
    "HardConstraintStrategy",
    "RerankerStrategyConfig",
    "SimpleReranker",
    "rerank",
    "recommendations_from_ranking",
    "EvidenceCoverageReranker",
]
