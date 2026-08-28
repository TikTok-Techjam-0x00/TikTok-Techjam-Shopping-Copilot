"""Frontend-only step runner that calls the real pipeline components."""

from __future__ import annotations

import copy
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dialogue import decide_ask, record_asked_attribute
from src.reranking import SimpleReranker, recommendations_from_ranking
from src.retrieval import Retriever
from src.state import (
    ShoppingState,
    create_state,
    retrieval_query,
    sanitize_retrieval_text,
    update_state,
)

from frontend.serializers import candidate_view, json_safe, ranked_candidate_view


STAGES = ("input", "state", "query", "retrieval", "reranking", "dialogue", "response")
STAGE_LABELS = {
    "input": "Input",
    "state": "State Update",
    "query": "Retrieval Query",
    "retrieval": "Retrieval",
    "reranking": "Reranking",
    "dialogue": "Dialogue Decision",
    "response": "Final Response",
}
IMPLEMENTATIONS = {
    "input": "frontend.trace_runner.TraceRunner.start_turn",
    "state": "src.state.update_state",
    "query": "src.state.retrieval_query / sanitize_retrieval_text",
    "retrieval": "src.retrieval.Retriever.retrieve",
    "reranking": "src.reranking.SimpleReranker.rerank",
    "dialogue": "src.dialogue.decide_ask / record_asked_attribute",
    "response": "src.reranking.recommendations_from_ranking",
}


@dataclass
class TurnTrace:
    turn: int
    message: str
    user_profile: dict[str, Any]
    previous_asked_attribute: str | None
    top_k: int
    retrieval_k: int
    working_state: ShoppingState
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_candidates: list[Any] = field(default_factory=list)
    raw_ranked: list[Any] = field(default_factory=list)
    query: str = ""
    dialogue: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    committed: bool = False


@dataclass
class DeveloperSession:
    session_id: str
    user_profile: dict[str, Any]
    committed_state: ShoppingState
    turn: int = 0
    last_asked: str | None = None
    active: TurnTrace | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class TraceRunner:
    """Execute existing pipeline components once per stage and retain traces."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = Retriever.bm25(str(self.catalog_path))
        self.reranker = SimpleReranker()
        self.sessions: dict[str, DeveloperSession] = {}

    def create_session(self, user_profile: dict[str, Any] | None = None) -> dict[str, Any]:
        profile = dict(user_profile or {})
        session_id = f"dev_{uuid.uuid4().hex}"
        self.sessions[session_id] = DeveloperSession(
            session_id=session_id,
            user_profile=profile,
            committed_state=create_state(session_id, profile),
        )
        return self.session_view(session_id)

    def _session(self, session_id: str) -> DeveloperSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError("Developer session not found") from exc

    def start_turn(
        self,
        session_id: str,
        message: str,
        *,
        top_k: int = 10,
        retrieval_k: int = 100,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        if session.active and not session.active.committed:
            raise RuntimeError("Finish, commit, or restart the active developer turn first")
        if session.turn >= 10:
            raise RuntimeError("Developer session has reached the 10-turn limit")
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("User message must not be empty")
        if top_k < 1 or top_k > 10:
            raise ValueError("top_k must be between 1 and 10")
        if retrieval_k < top_k or retrieval_k > 100:
            raise ValueError("retrieval_k must be between top_k and 100")

        trace = TurnTrace(
            turn=session.turn + 1,
            message=clean_message,
            user_profile=dict(session.user_profile),
            previous_asked_attribute=session.last_asked,
            top_k=top_k,
            retrieval_k=retrieval_k,
            working_state=session.committed_state.copy(),
        )
        trace.stages["input"] = {
            "status": "completed",
            "label": STAGE_LABELS["input"],
            "implementation": IMPLEMENTATIONS["input"],
            "duration_ms": 0.0,
            "input": {
                "session_id": session_id,
                "turn": trace.turn,
                "user_profile": trace.user_profile,
                "user_message": clean_message,
                "previous_asked_attribute": session.last_asked,
            },
            "output": {"message_stored": True},
        }
        session.active = trace
        return self.trace_view(session_id)

    def restart_turn(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        if session.active is None:
            raise RuntimeError("No active developer turn to restart")
        previous = session.active
        session.active = None
        return self.start_turn(
            session_id,
            previous.message,
            top_k=previous.top_k,
            retrieval_k=previous.retrieval_k,
        )

    def next_pending_stage(self, session_id: str) -> str | None:
        trace = self._active(session_id)
        for stage in STAGES[1:]:
            if trace.stages.get(stage, {}).get("status") != "completed":
                return stage
        return None

    def run_stage(self, session_id: str, stage: str) -> dict[str, Any]:
        trace = self._active(session_id)
        if stage not in STAGES[1:]:
            raise ValueError(f"Unknown executable stage: {stage}")
        existing = trace.stages.get(stage)
        if existing and existing.get("status") == "completed":
            return self.trace_view(session_id, selected_stage=stage)
        expected = self.next_pending_stage(session_id)
        if expected != stage:
            raise RuntimeError(f"Next valid stage is {expected or 'none'}, not {stage}")

        trace.stages[stage] = {
            "status": "running",
            "label": STAGE_LABELS[stage],
            "implementation": IMPLEMENTATIONS[stage],
        }
        started = time.perf_counter()
        try:
            runner = getattr(self, f"_run_{stage}")
            inputs, output = runner(trace)
            trace.stages[stage].update({
                "status": "completed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "input": json_safe(inputs),
                "output": json_safe(output),
            })
        except Exception as exc:
            trace.stages[stage].update({
                "status": "error",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "technical_details": traceback.format_exc(),
                },
            })
            raise
        return self.trace_view(session_id, selected_stage=stage)

    def run_next(self, session_id: str) -> dict[str, Any]:
        stage = self.next_pending_stage(session_id)
        if stage is None:
            return self.trace_view(session_id, selected_stage="response")
        return self.run_stage(session_id, stage)

    def run_all(self, session_id: str) -> dict[str, Any]:
        while (stage := self.next_pending_stage(session_id)) is not None:
            self.run_stage(session_id, stage)
        return self.trace_view(session_id, selected_stage="response")

    def commit_turn(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        trace = self._active(session_id)
        if trace.response is None or trace.stages.get("response", {}).get("status") != "completed":
            raise RuntimeError("Build the final response before committing the turn")
        if trace.committed:
            return self.session_view(session_id)
        trace.committed = True
        session.committed_state = trace.working_state.copy()
        session.turn = trace.turn
        session.last_asked = trace.dialogue.get("ask_attribute") if trace.dialogue else None
        session.history.append(self._serialize_trace(trace, selected_stage="response"))
        return self.session_view(session_id)

    def history_trace(self, session_id: str, turn: int) -> dict[str, Any]:
        session = self._session(session_id)
        for trace in session.history:
            if trace["turn"] == turn:
                return copy.deepcopy(trace)
        if session.active and session.active.turn == turn:
            return self.trace_view(session_id)
        raise KeyError(f"Developer turn {turn} not found")

    def _active(self, session_id: str) -> TurnTrace:
        session = self._session(session_id)
        if session.active is None:
            raise RuntimeError("Start a developer turn first")
        return session.active

    def _run_state(self, trace: TurnTrace) -> tuple[dict[str, Any], dict[str, Any]]:
        before = trace.working_state.to_dict()
        state = update_state(
            trace.working_state,
            trace.message,
            turn=trace.turn,
            asked_attribute=trace.previous_asked_attribute,
        )
        trace.working_state = state
        return {
            "state_before": before,
            "user_message": trace.message,
            "turn": trace.turn,
            "asked_attribute": trace.previous_asked_attribute,
        }, {"state_before": before, "state_after": state.to_dict()}

    def _run_query(self, trace: TurnTrace) -> tuple[dict[str, Any], dict[str, Any]]:
        state_query = retrieval_query(trace.working_state)
        fallback_used = not bool(state_query)
        trace.query = state_query or sanitize_retrieval_text(trace.message)
        return {"state": trace.working_state.to_dict()}, {
            "query": trace.query,
            "source": "sanitize_retrieval_text(user_message)" if fallback_used else "retrieval_query(state)",
            "fallback_used": fallback_used,
        }

    def _run_retrieval(self, trace: TurnTrace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace.raw_candidates = self.retriever.retrieve(
            trace.query,
            state=trace.working_state,
            intent=trace.working_state.intent,
            k=trace.retrieval_k,
        )
        return {
            "query": trace.query,
            "state": trace.working_state.to_dict(),
            "intent": trace.working_state.intent,
            "k": trace.retrieval_k,
        }, [candidate_view(candidate) for candidate in trace.raw_candidates]

    def _run_reranking(self, trace: TurnTrace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace.raw_ranked = self.reranker.rerank(
            trace.working_state,
            trace.raw_candidates,
            top_k=trace.top_k,
        )
        return {
            "state": trace.working_state.to_dict(),
            "candidate_count": len(trace.raw_candidates),
            "top_k": trace.top_k,
        }, [ranked_candidate_view(candidate) for candidate in trace.raw_ranked]

    def _run_dialogue(self, trace: TurnTrace) -> tuple[dict[str, Any], dict[str, Any]]:
        trace.dialogue = dict(decide_ask(trace.working_state, trace.raw_candidates))
        record_asked_attribute(trace.working_state, trace.dialogue.get("ask_attribute"))
        return {
            "state": trace.working_state.to_dict(),
            "candidate_count": len(trace.raw_candidates),
        }, {**trace.dialogue, "state_after_record": trace.working_state.to_dict()}

    def _run_response(self, trace: TurnTrace) -> tuple[dict[str, Any], dict[str, Any]]:
        assert trace.dialogue is not None
        trace.response = {
            "message": trace.dialogue.get("message") or "Here are the closest matches I found.",
            "ask_attribute": trace.dialogue.get("ask_attribute"),
            "recommendations": recommendations_from_ranking(trace.raw_ranked, trace.top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        return {
            "dialogue_decision": trace.dialogue,
            "ranked_candidate_count": len(trace.raw_ranked),
            "top_k": trace.top_k,
        }, trace.response

    def _serialize_trace(self, trace: TurnTrace, selected_stage: str | None = None) -> dict[str, Any]:
        stages = []
        for name in STAGES:
            data = copy.deepcopy(trace.stages.get(name, {}))
            stages.append({
                "name": name,
                "label": STAGE_LABELS[name],
                "implementation": IMPLEMENTATIONS[name],
                "status": data.get("status", "not_run"),
                "duration_ms": data.get("duration_ms"),
                "input": data.get("input"),
                "output": data.get("output"),
                "error": data.get("error"),
            })
        return {
            "turn": trace.turn,
            "message": trace.message,
            "committed": trace.committed,
            "selected_stage": selected_stage or self.next_stage_from_trace(trace),
            "stages": stages,
            "parameters": {
                "session_id": trace.working_state.session_id,
                "turn": trace.turn,
                "top_k": trace.top_k,
                "retrieval_k": trace.retrieval_k,
                "catalog_path": str(self.catalog_path),
                "retriever_type": type(self.retriever.strategy).__name__,
                "reranker_type": type(self.reranker).__name__,
            },
        }

    @staticmethod
    def next_stage_from_trace(trace: TurnTrace) -> str:
        completed = [name for name in STAGES if trace.stages.get(name, {}).get("status") == "completed"]
        return completed[-1] if completed else "input"

    def trace_view(self, session_id: str, selected_stage: str | None = None) -> dict[str, Any]:
        return self._serialize_trace(self._active(session_id), selected_stage)

    def session_view(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        return {
            "session_id": session.session_id,
            "turn": session.turn,
            "last_asked_attribute": session.last_asked,
            "user_profile": session.user_profile,
            "committed_state": session.committed_state.to_dict(),
            "history": [
                {"turn": trace["turn"], "message": trace["message"]}
                for trace in session.history
            ],
            "active_trace": self._serialize_trace(session.active) if session.active else None,
        }
