"""Deterministic Buying/Browsing intent router."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Intent = Literal["buying", "browsing", "unknown"]


@dataclass(frozen=True, slots=True)
class IntentResult:
    intent: Intent
    confidence: float
    is_override: bool = False
    evidence: tuple[str, ...] = ()
    is_conflict: bool = False

    @property
    def is_clear(self) -> bool:
        """Whether the high-precision rule layer can be used directly."""

        return self.intent != "unknown" and not self.is_conflict and self.confidence >= 0.70


BUYING_PATTERNS = {
    "explicit purchase": r"\b(?:buy|purchase|order)\b",
    "direct search": r"\b(?:find|show|recommend|get) me\b",
    "stated need": r"\b(?:i need|i want|looking for|need a|want a)\b",
    "hard constraint": r"\b(?:under|below|less than|no more than)\s*[$£€]?\s*\d+",
    "comparison": r"\b(?:which|what) .{0,30}(?:should i buy|is better)\b",
}
BROWSING_PATTERNS = {
    "explicit browsing": r"\b(?:just browsing|just looking|browse|still exploring)\b",
    "ideas or inspiration": r"\b(?:ideas?|inspiration|trending|popular)\b",
    "category exploration": r"\b(?:what kinds|what types|what options|what is available)\b",
    "open exploration": r"\b(?:show me something|surprise me)\b",
}
OVERRIDE_PATTERNS = {
    "explicit correction": r"\b(?:actually|instead|rather than|change that to)\b",
    "discard old request": r"\b(?:forget|never mind|scratch)\b",
    "new intent": r"\b(?:now i need|now i want)\b",
}
CONSTRAINT_ONLY_PATTERN = re.compile(
    r"(?:[$£€]\s*\d+|\b\d+\s*(?:dollars?|usd|pounds?|euros?)\b|"
    r"\b(?:size|color|colour|brand|material|black|white|blue|red|small|medium|large)\b)",
    re.IGNORECASE,
)


def _matches(text: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items() if re.search(pattern, text, re.I)]


def classify_intent(message: str, previous_intent: Intent = "unknown") -> IntentResult:
    text = " ".join(message.strip().split())
    if not text:
        return IntentResult("unknown", 0.0, evidence=("empty message",))

    buying = _matches(text, BUYING_PATTERNS)
    browsing = _matches(text, BROWSING_PATTERNS)
    override = _matches(text, OVERRIDE_PATTERNS)
    buying_score = len(buying) * 2
    browsing_score = len(browsing) * 2

    # An explicit browsing phrase is an intentional override of generic search
    # language such as "looking for" and remains a high-precision rule result.
    if buying and "explicit browsing" in browsing:
        return IntentResult(
            "browsing",
            0.86,
            bool(override),
            tuple(browsing + buying + override),
        )

    # Mixed rule families are not resolved by score alone. The provisional
    # winner is retained for rule-only degradation, while the conflict flag
    # routes a configured semantic resolver through the validation path.
    if buying and browsing:
        provisional: Intent = previous_intent
        if buying_score > browsing_score:
            provisional = "buying"
        elif browsing_score > buying_score:
            provisional = "browsing"
        return IntentResult(
            provisional,
            0.45,
            bool(override),
            tuple(buying + browsing + override),
            is_conflict=True,
        )

    if buying_score > browsing_score:
        return IntentResult("buying", min(0.98, 0.72 + 0.08 * len(buying)), bool(override), tuple(buying + override))
    if browsing_score > buying_score:
        return IntentResult("browsing", min(0.98, 0.72 + 0.08 * len(browsing)), bool(override), tuple(browsing + override))
    if previous_intent != "unknown" and CONSTRAINT_ONLY_PATTERN.search(text):
        return IntentResult(previous_intent, 0.78, bool(override), ("inherited from previous intent", *override))
    if previous_intent != "unknown" and not override:
        return IntentResult(previous_intent, 0.60, evidence=("conversation context",))
    return IntentResult("unknown", 0.35, bool(override), tuple(override) or ("no clear intent signal",))
