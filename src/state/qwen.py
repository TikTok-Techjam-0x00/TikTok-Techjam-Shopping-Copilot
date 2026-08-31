"""Qwen semantic resolver configured through local environment variables."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from .semantic import CallableSemanticResolver, SemanticRequest, SemanticResolution


DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"

SYSTEM_PROMPT = """You extract shopping intent and constraints from English conversation context.
Return one JSON object only. Never return prose or markdown.

Use this schema:
{
  "intent": "buying" | "browsing" | null,
  "hard_constraint": {"category|material|color|size|style|brand|budget|feature|use_case|other": value},
  "soft_constraint": {"category|material|color|size|style|brand|budget|feature|use_case|other": value},
  "no_preference": [attribute_name],
  "rejected_values": {attribute_name: value},
  "override": boolean,
  "clear_hard_constraint": boolean,
  "clear_soft_constraint": boolean,
  "confidence": number
}

Only include facts supported by the message and supplied context. Use English values only.
Do not invent products, brands, prices, or preferences. Explicit facts extracted by rules
are authoritative; focus on references, comparisons, alternatives, and contextual meaning
that the rules did not resolve.
"""


class QwenSemanticResolver:
    """Resolve ambiguous state updates through Qwen's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DASHSCOPE_API_KEY must not be empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
        self.client = client
        self._normalizer = CallableSemanticResolver(lambda request: self._call(request))

    def reset_usage(self) -> None:
        """Reset provider token counts for the next Agent turn."""

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def model_usage(self) -> tuple[int, int]:
        """Return provider-reported input and output tokens since reset."""

        return self.prompt_tokens, self.completion_tokens

    @classmethod
    def from_env(cls) -> QwenSemanticResolver | None:
        """Build a resolver when a local API key is configured, otherwise disable it."""

        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).strip()
        model = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL).strip()
        raw_timeout = os.getenv("QWEN_API_TIMEOUT_SECONDS", "30").strip()
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError:
            timeout_seconds = 30.0
        raw_retries = os.getenv("QWEN_API_MAX_RETRIES", "3").strip()
        try:
            max_retries = int(raw_retries)
        except ValueError:
            max_retries = 3
        return cls(
            api_key=api_key,
            base_url=base_url or DEFAULT_QWEN_BASE_URL,
            model=model or DEFAULT_QWEN_MODEL,
            timeout_seconds=max(1.0, timeout_seconds),
            max_retries=max(0, max_retries),
        )

    def resolve(self, request: SemanticRequest) -> SemanticResolution | None:
        return self._normalizer.resolve(request)

    def _call(self, request: SemanticRequest) -> dict[str, Any] | None:
        payload = {
            "message": request.message,
            "current_state": request.current_state,
            "recent_history": list(request.recent_history),
            "asked_attribute": request.asked_attribute,
            "rule_result": {
                "intent": request.rule_intent,
                "intent_confidence": request.rule_intent_confidence,
                "evidence": list(request.rule_evidence),
            },
            "fallback_reasons": list(request.fallback_reasons),
        }
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        response = self.client.chat.completions.create(**request_body)
        usage = getattr(response, "usage", None)
        self.prompt_tokens += max(
            0,
            int(getattr(usage, "prompt_tokens", 0) or 0),
        )
        self.completion_tokens += max(
            0,
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            return None
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None

def qwen_semantic_resolver_from_env() -> QwenSemanticResolver | None:
    """Return the configured Qwen resolver without requiring secrets in code."""

    return QwenSemanticResolver.from_env()
