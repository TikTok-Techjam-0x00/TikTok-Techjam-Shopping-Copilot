"""Offline evidence-coverage reranking for disclosed catalog constraints."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..attribute import AttributeName, AttributeValue
from ..item import Candidate, Candidates10, RankedCandidate


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_FRAGMENT_SEPARATOR_RE = re.compile(r"\s*;\s*|[\r\n]+")
_PHRASE_EVIDENCE_WEIGHT = 0.25
_STOPWORDS = frozenset({
    "a", "an", "and", "at", "color", "for", "from", "in",
    "is", "it", "matters", "of", "on", "or", "product", "that", "the",
    "this", "to", "what", "with",
})


def _state_value(state: object, field: str, default: Any = None) -> Any:
    return state.get(field, default) if isinstance(state, Mapping) else getattr(state, field, default)


def _ordered_tokens(
    value: object,
    *,
    keep_single_character: bool = False,
) -> tuple[str, ...]:
    return tuple(
        token.casefold()
        for token in _TOKEN_RE.findall(str(value))
        if (len(token) > 1 or keep_single_character)
        and token.casefold() not in _STOPWORDS
    )


def _tokens(value: object) -> frozenset[str]:
    return frozenset(_ordered_tokens(value))


def _normalized_phrase(value: object) -> str:
    # One-character terms are noisy as unordered terms, but meaningful inside
    # ordered phrases such as ``5% spandex``, ``8 inch``, or ``size M``.
    return " ".join(_ordered_tokens(value, keep_single_character=True))


def _flatten(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, AttributeValue):
        return [
            *value.values,
            *(item for values in value.details.values() for item in values),
        ]
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_flatten(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [str(value)]


def _attribute_name(value: object) -> str:
    return str(getattr(value, "value", value)).strip().casefold()


def _record_epoch(state: object, group: str, attribute: object) -> int | None:
    provenance = _state_value(state, "constraint_provenance", {})
    if not isinstance(provenance, Mapping):
        return None
    records = provenance.get(group, {})
    if not isinstance(records, Mapping):
        return None
    record = records.get(attribute, records.get(_attribute_name(attribute)))
    if record is None:
        return None
    value = record.get("constraint_epoch") if isinstance(record, Mapping) else getattr(record, "constraint_epoch", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _active_fragments(state: object) -> list[tuple[str, float, str]]:
    """Return active constraint texts as ``(text, weight, attribute)``."""

    current_epoch = int(_state_value(state, "constraint_epoch", 0) or 0)
    fragments: list[tuple[str, float, str]] = []
    for group_name, base_weight in (("hard_constraint", 3.0), ("soft_constraint", 2.0)):
        constraints = _state_value(state, group_name, {})
        if not isinstance(constraints, Mapping):
            continue
        for attribute, raw_value in constraints.items():
            name = _attribute_name(attribute)
            # An override supersedes old soft preferences even when State keeps
            # them for diagnostics and audit history.
            epoch = _record_epoch(state, group_name, attribute)
            if group_name == "soft_constraint" and current_epoch > 0 and epoch != current_epoch:
                continue
            weight = 1.0 if name == AttributeName.CATEGORY.value else base_weight
            for text in _flatten(raw_value):
                # One customer reply can contain two independent requirements
                # separated by a semicolon. Score them as atomic evidence so a
                # candidate cannot receive full credit for matching only one.
                for fragment in _FRAGMENT_SEPARATOR_RE.split(text):
                    if _tokens(fragment):
                        fragments.append((fragment, weight, name))
    return fragments


def _product_text(candidate: Candidate) -> str:
    item = candidate.item
    return " ".join(
        _flatten(item.title)
        + _flatten(item.categories)
        + _flatten(item.features)
        + [str(item.details)]
        + _flatten(item.description)
        + _flatten(item.store)
    )


def _product_evidence_units(candidate: Candidate) -> frozenset[str]:
    """Return normalized catalog fields usable as exact evidence tie-breaks."""

    item = candidate.item
    values: list[object] = [
        item.title,
        *item.categories,
        *item.features,
        *item.description,
        item.store,
    ]
    for key, value in item.details.items():
        values.extend((value, f"{key} {value}"))
    return frozenset(
        phrase
        for value in values
        if (phrase := _normalized_phrase(value))
    )


class EvidenceCoverageReranker:
    """Rank candidates by IDF-weighted recall of disclosed requirements.

    The simulator's requirements are copied from catalog evidence.  Measuring
    how completely each candidate supports those words therefore gives a much
    sharper offline signal than treating every BM25 term as an independent OR.
    Candidate-local IDF prevents generic words such as ``women`` or ``cotton``
    from overpowering distinctive feature text.
    """

    def rerank(
        self,
        shopping_state: object,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> Candidates10:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")

        candidates: list[Candidate] = []
        seen: set[str] = set()
        for raw in candidates_100[:100]:
            try:
                candidate = raw if isinstance(raw, Candidate) else Candidate.from_dict(raw)
            except (TypeError, ValueError):
                continue
            if not candidate.parent_asin or candidate.parent_asin in seen:
                continue
            seen.add(candidate.parent_asin)
            candidates.append(candidate)
        if not candidates:
            return []

        fragments = _active_fragments(shopping_state)
        if not fragments:
            return [
                RankedCandidate.from_candidate(
                    candidate,
                    rerank_rank=rank,
                    rerank_score=round(1.0 / rank, 6),
                    matched=[],
                    violation=[],
                )
                for rank, candidate in enumerate(candidates[:top_k], start=1)
            ]

        product_texts = [_product_text(candidate) for candidate in candidates]
        product_tokens = [_tokens(text) for text in product_texts]
        product_phrases = [_normalized_phrase(text) for text in product_texts]
        product_units = [_product_evidence_units(candidate) for candidate in candidates]
        query_tokens = set().union(*(_tokens(text) for text, _, _ in fragments))
        inverse_frequency = {
            token: math.log(
                (len(candidates) + 1)
                / (1 + sum(token in observed for observed in product_tokens))
            ) + 1.0
            for token in query_tokens
        }
        fragment_phrases = [
            _normalized_phrase(text)
            for text, _, _ in fragments
        ]
        phrase_frequency = [
            sum(phrase in product for product in product_phrases)
            for phrase in fragment_phrases
        ]

        scored: list[
            tuple[float, float, float, float, float, int, Candidate, list[str]]
        ] = []
        for index, (candidate, observed, product_phrase, evidence_units) in enumerate(
            zip(candidates, product_tokens, product_phrases, product_units)
        ):
            weighted_coverage = 0.0
            phrase_evidence = 0.0
            exact_unit_evidence = 0.0
            total_weight = 0.0
            complete_matches = 0
            matched_attributes: list[str] = []
            for fragment_index, (text, fragment_weight, attribute) in enumerate(fragments):
                requested = _tokens(text)
                denominator = sum(inverse_frequency[token] for token in requested)
                if denominator <= 0:
                    continue
                coverage = sum(
                    inverse_frequency[token]
                    for token in requested
                    if token in observed
                ) / denominator
                weighted_coverage += fragment_weight * coverage
                total_weight += fragment_weight
                normalized = fragment_phrases[fragment_index]
                if normalized and normalized in product_phrase:
                    phrase_idf = math.log(
                        (len(candidates) + 1)
                        / (1 + phrase_frequency[fragment_index])
                    ) + 1.0
                    phrase_evidence += fragment_weight * phrase_idf
                if normalized and normalized in evidence_units:
                    # Exact catalog-field equality is only a tie-breaker. This
                    # keeps robust token/phrase recall primary while preferring
                    # direct evidence over words scattered across long text.
                    exact_unit_evidence += fragment_weight
                if coverage >= 0.999999:
                    complete_matches += 1
                    if attribute not in matched_attributes:
                        matched_attributes.append(attribute)
            score = (
                (
                    weighted_coverage
                    + _PHRASE_EVIDENCE_WEIGHT * phrase_evidence
                )
                / total_weight
                if total_weight
                else 0.0
            )
            popularity = math.log1p(candidate.item.rating_number or 0)
            average_rating = float(candidate.item.average_rating or 0.0)
            retrieval_tiebreak = float(candidate.retrieval_score or 0.0)
            scored.append(
                (
                    score,
                    exact_unit_evidence,
                    popularity,
                    average_rating,
                    retrieval_tiebreak,
                    -index,
                    candidate,
                    matched_attributes,
                )
            )

        # Rank at the same precision exposed by ``rerank_score``. Sub-machine
        # floating differences should not bypass the explicit evidence and
        # stability tie-breakers that follow the primary score.
        scored.sort(
            key=lambda entry: (round(entry[0], 6), *entry[1:6]),
            reverse=True,
        )
        return [
            RankedCandidate.from_candidate(
                candidate,
                rerank_rank=rank,
                rerank_score=round(score, 6),
                matched=matched,
                violation=[],
            )
            for rank, (score, _, _, _, _, _, candidate, matched) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]

    def rank_all(
        self,
        shopping_state: object,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
    ) -> list[RankedCandidate]:
        """Return the complete local order for late-turn exploration."""

        return list(self.rerank(shopping_state, candidates_100, top_k=100))


__all__ = ["EvidenceCoverageReranker"]
