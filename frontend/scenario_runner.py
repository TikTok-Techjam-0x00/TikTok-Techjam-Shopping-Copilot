"""Frontend-only controller for visualizing the official evaluator conversation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from src.retrieval.catalog import Catalog


@dataclass
class ScenarioSession:
    sample: dict[str, Any]
    effective_sample: dict[str, Any]
    target: str
    user_message: str
    disclosed: set[str] = field(default_factory=set)
    boundary_used: bool = False
    override_applied: bool = True
    turn: int = 0
    done: bool = False
    hit: bool = False
    hit_rank: int | None = None
    stop_reason: str | None = None


class ScenarioRunner:
    """Use evaluator helpers to prepare messages without exposing labels to Agent."""

    def __init__(self, dataset_path: str | Path, catalog: Catalog) -> None:
        self.dataset_path = Path(dataset_path)
        self.samples = load_jsonl(self.dataset_path)
        self.samples_by_id = {str(sample["sample_id"]): sample for sample in self.samples}
        self.catalog = catalog
        self.sessions: dict[str, ScenarioSession] = {}

    def sample_list(self) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": str(sample["sample_id"]),
                "scenario_type": str(sample["scenario_type"]),
                "profile_summary": str(sample.get("user_profile", {}).get("summary") or ""),
            }
            for sample in self.samples
        ]

    def start(self, session_id: str, sample_id: str) -> dict[str, Any]:
        if sample_id not in self.samples_by_id:
            raise KeyError(f"Public sample not found: {sample_id}")
        sample = self.samples_by_id[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, self.catalog)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        message = initial_message(
            effective_sample,
            coarse_category(list(self.catalog[target].categories)),
            disclosed,
        )
        scenario = ScenarioSession(
            sample=sample,
            effective_sample=effective_sample,
            target=target,
            user_message=message,
            disclosed=disclosed,
            override_applied=sample["scenario_type"] != "intent_override",
        )
        self.sessions[session_id] = scenario
        return self.view(session_id)

    def current_message(self, session_id: str) -> str:
        scenario = self._session(session_id)
        if scenario.done:
            raise RuntimeError("Evaluator scenario is already complete")
        return scenario.user_message

    def advance(self, session_id: str, response: dict[str, Any]) -> dict[str, Any]:
        scenario = self._session(session_id)
        if scenario.done:
            raise RuntimeError("Evaluator scenario is already complete")

        scenario.turn += 1
        recommendations = response.get("recommendations")
        ranked = [
            str(item.get("parent_asin"))
            for item in recommendations or []
            if isinstance(item, dict) and item.get("parent_asin")
        ]
        if scenario.override_applied and scenario.target in ranked:
            scenario.hit = True
            scenario.hit_rank = ranked.index(scenario.target) + 1
            scenario.done = True
            scenario.stop_reason = "target_hit"
            return self.view(session_id)

        if scenario.turn >= MAX_TURNS:
            scenario.done = True
            scenario.stop_reason = "max_turns"
            return self.view(session_id)

        override = scenario.effective_sample.get("behavior", {}).get("override") or {}
        if not scenario.override_applied and scenario.turn + 1 == int(override.get("turn", 3)):
            scenario.override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                scenario.disclosed.add(new_value)
            scenario.user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            scenario.user_message, scenario.boundary_used = customer_reply(
                scenario.effective_sample,
                response.get("ask_attribute"),
                scenario.disclosed,
                scenario.boundary_used,
            )
        return self.view(session_id)

    def view(self, session_id: str) -> dict[str, Any]:
        scenario = self._session(session_id)
        return {
            "sample_id": str(scenario.sample["sample_id"]),
            "scenario_type": str(scenario.sample["scenario_type"]),
            "turn": scenario.turn,
            "next_user_message": None if scenario.done else scenario.user_message,
            "done": scenario.done,
            "hit": scenario.hit,
            "hit_rank": scenario.hit_rank,
            "stop_reason": scenario.stop_reason,
            "override_applied": scenario.override_applied,
            "max_turns": MAX_TURNS,
            "user_profile": dict(scenario.sample.get("user_profile") or {}),
        }

    def _session(self, session_id: str) -> ScenarioSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError("Evaluator scenario session not found") from exc
