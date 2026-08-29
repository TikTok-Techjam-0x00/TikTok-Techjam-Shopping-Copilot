"""Pure Qwen reranking experiment with a retrieval-order safety fallback."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.item import Candidate, Candidates10, RankedCandidate
from src.reranking import EvidenceCoverageReranker


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = """You are the final Candidates100-to-Candidates10 reranker for a multi-turn shopping copilot.
Optimize the ordered TopK for target-product Hit@10 first and reciprocal rank second. Rank only the supplied candidates.

CONVERSATION INTERPRETATION
- USER messages are the only authority for needs and preferences. AGENT messages provide question context or suggestions; never treat an option mentioned only by the AGENT as accepted.
- Read the conversation chronologically. Resolve short USER replies such as "black", "any is fine", or "for that..." against the immediately preceding AGENT question.
- The newest explicit correction or override defines the active shopping request. Discard older category, use-case, or preference statements that conflict with it; retain older facts only when they remain compatible.
- "No preference", "doesn't matter", "use your judgment", and equivalent replies remove that attribute from consideration. They are not negative product constraints.
- "No/avoid/without X" is a negative constraint. Do not reward candidates merely for containing the word X.

FIT EVIDENCE
- Infer the active product category, must-have constraints, rejected values, budget/ranges, and optional preferences from USER messages before comparing products.
- Use only candidate evidence: title, categories, features, details, price, store, average_rating, and rating_number. Missing evidence is unknown, not a match. Never fabricate attributes.
- Interpret obvious wording variants and category synonyms semantically, but do not relax an explicit numeric limit or a clear rejection.
- For budget limits, rank products within the limit above products over the limit when price is known. Unknown price is less certain than a confirmed in-budget price, but is not automatically a violation.

LEXICOGRAPHIC RANKING PRIORITY
1. Active category and intended use compatibility.
2. Number and importance of satisfied explicit must-have constraints; demote clear violations strongly.
3. Satisfaction of the most recent and most specific USER requirements.
4. Satisfaction of optional preferences, without over-penalizing missing metadata.
5. Product quality signals when relevant and sufficiently supported by rating count.
6. retrieval_rank and retrieval_score only as final tie-breakers between similarly fitting products.

For exploratory/browsing language, keep all TopK items strongly category-relevant, then provide useful variation across plausible styles or features. For a concrete buying request, prioritize exact constraint satisfaction over variety.

Return exactly top_k unique parent_asin strings when enough candidates are supplied. Every returned ID must occur in candidates. Output only one strict JSON array ordered best-first. Do not output prose, markdown, scores, reasons, analysis, or chain-of-thought."""


def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return str(value)


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    item = candidate.item
    return {
        "parent_asin": item.parent_asin,
        "title": item.title,
        "price": item.price,
        "categories": list(item.categories),
        "features": list(item.features)[:12],
        "details": dict(list(item.details.items())[:20]),
        "store": item.store,
        "average_rating": item.average_rating,
        "rating_number": item.rating_number,
        "retrieval_rank": candidate.retrieval_rank,
        "retrieval_score": candidate.retrieval_score,
    }


def _unique_candidates(values: Sequence[Candidate | Mapping[str, Any]]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    for value in values[:100]:
        try:
            candidate = value if isinstance(value, Candidate) else Candidate.from_dict(value)
        except (TypeError, ValueError):
            continue
        if not candidate.parent_asin or candidate.parent_asin in seen:
            continue
        seen.add(candidate.parent_asin)
        result.append(candidate)
    return result


def _parse_parent_asins(content: object) -> list[str]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen response content is empty")
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.I | re.S)
    if fenced:
        cleaned = fenced.group(1)
    parsed = json.loads(cleaned)
    if isinstance(parsed, Mapping):
        parsed = next(
            (parsed.get(key) for key in ("parent_asins", "ranking", "ranked_parent_asins") if key in parsed),
            None,
        )
    if not isinstance(parsed, list):
        raise ValueError("Qwen response must be a JSON array of parent_asin strings")
    return [str(value).strip() for value in parsed if isinstance(value, str) and value.strip()]


class QwenReranker:
    """Rerank retrieval candidates using Qwen without changing core models."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        use_local_fallback: bool = False,
    ) -> None:
        load_dotenv(REPOSITORY_ROOT / ".env")
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        self.base_url = os.getenv("QWEN_BASE_URL", "").strip()
        self.model = os.getenv("QWEN_MODEL", "qwen-plus").strip() or "qwen-plus"
        self._provided_client = client
        self.use_local_fallback = bool(use_local_fallback)
        self.local_fallback = EvidenceCoverageReranker()
        self._conversation: list[dict[str, str]] = []
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def set_conversation(self, messages: Sequence[Mapping[str, object]]) -> None:
        """Store an isolated copy of all observable USER/AGENT messages."""
        conversation: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                conversation.append({"role": role, "content": content})
        self._conversation = conversation

    def _client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        if not self.base_url:
            raise RuntimeError("QWEN_BASE_URL is not configured")
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=None,
            max_retries=0,
        )

    def _rank_with_qwen(self, candidates: list[Candidate], top_k: int) -> list[str]:
        prompt = {
            "task": "Select and order the products most likely to satisfy the user's current active request.",
            "metric_priority": ["Hit@10", "reciprocal_rank"],
            "top_k": top_k,
            "conversation_order": "oldest_to_newest",
            "conversation": list(self._conversation),
            "candidates": [_candidate_payload(candidate) for candidate in candidates],
            "output_contract": {
                "type": "json_array",
                "items": "parent_asin from candidates",
                "unique": True,
                "length": top_k,
                "order": "best_first",
            },
        }
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))},
            ],
            temperature=0,
        )
        usage = getattr(response, "usage", None)
        self.last_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        choices = getattr(response, "choices", None) or []
        content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
        return _parse_parent_asins(content)

    def rerank(
        self,
        shopping_state: object,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> Candidates10:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        candidates = _unique_candidates(candidates_100)
        if not candidates:
            return []

        requested_order: list[str] = []
        try:
            requested_order = self._rank_with_qwen(candidates, top_k)
        except Exception:
            requested_order = []

        if not requested_order and self.use_local_fallback:
            return self.local_fallback.rerank(
                shopping_state,
                candidates,
                top_k=top_k,
            )

        by_asin = {candidate.parent_asin: candidate for candidate in candidates}
        selected: list[Candidate] = []
        seen: set[str] = set()
        for parent_asin in requested_order:
            candidate = by_asin.get(parent_asin)
            if candidate is None or parent_asin in seen:
                continue
            selected.append(candidate)
            seen.add(parent_asin)
            if len(selected) >= top_k:
                break
        for candidate in candidates:
            if len(selected) >= top_k:
                break
            if candidate.parent_asin not in seen:
                selected.append(candidate)
                seen.add(candidate.parent_asin)

        return [
            RankedCandidate.from_candidate(
                candidate,
                rerank_rank=rank,
                rerank_score=round(1.0 / rank, 6),
                matched=[],
                violation=[],
            )
            for rank, candidate in enumerate(selected, start=1)
        ]
