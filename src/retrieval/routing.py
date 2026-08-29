"""Intent-aware routing between complete Retrieval strategies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ..item import Candidates100
from .retriever import RetrievalStrategy


RouteName = Literal["buying", "browsing"]


def _state_intent(state: object | None) -> object | None:
    if isinstance(state, Mapping):
        return state.get("intent")
    return getattr(state, "intent", None)


def _state_turn(state: object | None) -> int | None:
    value = state.get("turn") if isinstance(state, Mapping) else getattr(state, "turn", None)
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _intent_name(intent: object | None, state: object | None) -> str:
    value = intent if intent is not None else _state_intent(state)
    return str(getattr(value, "value", value) or "").strip().casefold()


@dataclass(frozen=True, slots=True)
class IntentRoutingConfig:
    """Routing policy; unknown intents use the conservative Buying strategy."""

    unknown_route: RouteName = "buying"
    browsing_max_turn: int | None = None

    def __post_init__(self) -> None:
        if self.unknown_route not in ("buying", "browsing"):
            raise ValueError("unknown_route must be 'buying' or 'browsing'")
        if self.browsing_max_turn is not None and self.browsing_max_turn <= 0:
            raise ValueError("browsing_max_turn must be positive when supplied")


class IntentRoutedRetriever:
    """Delegate one query to the strategy selected by current ShoppingState intent.

    The router never merges candidates or changes their scores.  It only chooses
    one already-valid Retrieval strategy, so the Candidate contract remains
    identical for 3A and 3B.
    """

    def __init__(
        self,
        buying: RetrievalStrategy,
        browsing: RetrievalStrategy,
        *,
        config: IntentRoutingConfig | None = None,
    ) -> None:
        self.buying = buying
        self.browsing = browsing
        self.config = config or IntentRoutingConfig()
        buying_catalog = getattr(buying, "catalog", None)
        browsing_catalog = getattr(browsing, "catalog", None)
        # Keep the existing integration/replay contract when both routes search
        # the same frozen catalog.  A mismatched pair deliberately exposes no
        # catalog rather than letting downstream code use the wrong product set.
        self.catalog = buying_catalog if buying_catalog is browsing_catalog else None

    def route_name(
        self,
        state: object | None = None,
        intent: object | None = None,
    ) -> RouteName:
        name = _intent_name(intent, state)
        if name == "browsing":
            turn = _state_turn(state)
            if (
                self.config.browsing_max_turn is not None
                and turn is not None
                and turn > self.config.browsing_max_turn
            ):
                return "buying"
            return "browsing"
        if name == "buying":
            return "buying"
        return self.config.unknown_route

    def retrieve(
        self,
        query: str | None,
        state: object | None = None,
        intent: str | None = None,
        k: int = 100,
    ) -> Candidates100:
        strategy = self.browsing if self.route_name(state, intent) == "browsing" else self.buying
        return strategy.retrieve(query, state, intent, k)


__all__ = ["IntentRoutingConfig", "IntentRoutedRetriever", "RouteName"]
