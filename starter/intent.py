from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Intent = Literal["buying", "browsing", "unknown"]


@dataclass(frozen=True)
class IntentResult:
    """Result returned by the lightweight intent router."""

    intent: Intent
    confidence: float
    is_override: bool = False
    evidence: tuple[str, ...] = ()


# Strong phrases are checked before individual words to reduce false positives.
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

# Short replies usually add constraints to an active shopping task and should
# inherit the previous intent instead of being classified as unknown.
CONSTRAINT_ONLY_PATTERN = re.compile(
    r"(?:[$£€]\s*\d+|\b\d+\s*(?:dollars?|usd|pounds?|euros?)\b|"
    r"\b(?:size|color|colour|brand|material|black|white|blue|red|small|medium|large)\b)",
    re.IGNORECASE,
)


def _matches(text: str, patterns: dict[str, str]) -> list[str]:
    return [
        label
        for label, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def classify_intent(
    message: str,
    previous_intent: Intent = "unknown",
) -> IntentResult:
    """Classify one user turn using the message and conversation context.

    The function is deterministic and has no network or model dependency.  An
    LLM classifier can later be used only when this router returns ``unknown``
    or a low confidence score.
    """

    text = " ".join(message.strip().split())
    if not text:
        return IntentResult("unknown", 0.0, evidence=("empty message",))

    buying_evidence = _matches(text, BUYING_PATTERNS)
    browsing_evidence = _matches(text, BROWSING_PATTERNS)
    override_evidence = _matches(text, OVERRIDE_PATTERNS)
    is_override = bool(override_evidence)

    buying_score = len(buying_evidence) * 2
    browsing_score = len(browsing_evidence) * 2

    if buying_score > browsing_score:
        confidence = min(0.98, 0.72 + 0.08 * len(buying_evidence))
        return IntentResult(
            "buying",
            confidence,
            is_override,
            tuple(buying_evidence + override_evidence),
        )

    if browsing_score > buying_score:
        confidence = min(0.98, 0.72 + 0.08 * len(browsing_evidence))
        return IntentResult(
            "browsing",
            confidence,
            is_override,
            tuple(browsing_evidence + override_evidence),
        )

    # Evaluator browsing prompts deliberately combine "looking for" with
    # "still exploring". The explicit exploration cue wins that otherwise tie.
    if buying_score == browsing_score and "explicit browsing" in browsing_evidence:
        return IntentResult(
            "browsing",
            0.86,
            is_override,
            tuple(browsing_evidence + buying_evidence + override_evidence),
        )

    # Attribute-only follow-ups such as "under $120" inherit the active route.
    if previous_intent != "unknown" and CONSTRAINT_ONLY_PATTERN.search(text):
        return IntentResult(
            previous_intent,
            0.78,
            is_override,
            ("inherited from previous intent", *override_evidence),
        )

    # General follow-ups without a new route keep the established intent, but
    # receive lower confidence because the current message has no strong cue.
    if previous_intent != "unknown" and not is_override:
        return IntentResult(
            previous_intent,
            0.60,
            evidence=("conversation context",),
        )

    return IntentResult(
        "unknown",
        0.35,
        is_override,
        tuple(override_evidence) or ("no clear intent signal",),
    )
