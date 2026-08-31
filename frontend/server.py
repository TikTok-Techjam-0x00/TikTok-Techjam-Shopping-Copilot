"""FastAPI adapter for the existing Shopping Copilot Agent.

This module owns only demo HTTP/session concerns and catalog enrichment. All
shopping intelligence remains in ``starter.Agent`` and the existing pipeline.
"""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


FRONTEND_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FRONTEND_DIR.parent
CATALOG_PATH = REPOSITORY_ROOT / "data" / "catalog.jsonl"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starter.agent import Agent  # noqa: E402
from src.retrieval.catalog import Catalog  # noqa: E402
from frontend.trace_runner import TraceRunner  # noqa: E402
from frontend.scenario_runner import ScenarioRunner  # noqa: E402


LOGGER = logging.getLogger("shopping_copilot.frontend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Shopping Copilot Demo", version="1.0.0")
catalog = Catalog.load(CATALOG_PATH)
agent = Agent(CATALOG_PATH, catalog=catalog)
trace_runner = TraceRunner(catalog)
scenario_runner = ScenarioRunner(REPOSITORY_ROOT / "data" / "public_set.jsonl", catalog)
sessions: dict[str, dict[str, int]] = {}
session_lock = threading.RLock()


class SessionRequest(BaseModel):
    user_profile: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class DeveloperTurnRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=10)
    retrieval_k: int = Field(default=100, ge=1, le=100)


class DeveloperActionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class DeveloperStageRequest(DeveloperActionRequest):
    stage: str = Field(min_length=1)


class EvaluatorSessionRequest(BaseModel):
    sample_id: str = Field(min_length=1)


class EvaluatorActionRequest(BaseModel):
    session_id: str = Field(min_length=1)


def _enrich_recommendations(recommendations: object) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    if not isinstance(recommendations, list):
        return enriched

    for rank, recommendation in enumerate(recommendations, start=1):
        if not isinstance(recommendation, dict):
            continue
        parent_asin = str(recommendation.get("parent_asin") or "").strip()
        if not parent_asin:
            continue
        product = catalog.get(parent_asin)
        if product is None:
            LOGGER.warning("Catalog product missing for parent_asin=%s", parent_asin)
            enriched.append({
                "rank": rank,
                "parent_asin": parent_asin,
                "product": None,
                "catalog_missing": True,
            })
            continue
        enriched.append({
            "rank": rank,
            "parent_asin": parent_asin,
            "product": product.to_dict(),
            "catalog_missing": False,
        })
    return enriched


def _with_evaluation_preview(session_id: str, trace: dict[str, Any]) -> dict[str, Any]:
    """Annotate a developer trace without mutating or informing the Agent."""
    scenario = scenario_runner.sessions.get(session_id)
    response_stage = next(
        (stage for stage in trace.get("stages", []) if stage.get("name") == "response"),
        None,
    )
    response = response_stage.get("output") if response_stage else None
    if scenario is None or not isinstance(response, dict):
        trace["evaluation_preview"] = {"status": "pending", "hit": False, "hit_rank": None}
        return trace
    ranked = [
        str(item.get("parent_asin"))
        for item in response.get("recommendations") or []
        if isinstance(item, dict) and item.get("parent_asin")
    ]
    hit = scenario.override_applied and scenario.target in ranked
    trace["evaluation_preview"] = {
        "status": "hit" if hit else "not_hit",
        "hit": hit,
        "hit_rank": ranked.index(scenario.target) + 1 if hit else None,
    }
    return trace


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "catalog_items": len(catalog),
        "catalog_source": catalog.source.name if catalog.source else None,
    }


@app.post("/api/session")
async def create_session(request: SessionRequest) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    try:
        with session_lock:
            agent.reset(session_id, request.user_profile)
            sessions[session_id] = {"turn": 0}
    except Exception as exc:  # pragma: no cover - defensive presentation boundary
        LOGGER.exception("Failed to create Agent session")
        raise HTTPException(status_code=500, detail="Unable to start a new shopping session.") from exc
    return {"session_id": session_id, "turn": 0}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Please enter a shopping message.")

    try:
        with session_lock:
            wrapper_state = sessions.get(request.session_id)
            if wrapper_state is None:
                raise HTTPException(status_code=404, detail="Session not found. Start a new session.")
            turn = wrapper_state["turn"] + 1
            if turn > 10:
                raise HTTPException(
                    status_code=409,
                    detail="This demo session has reached the 10-turn limit. Start a new session.",
                )

            response = agent.respond(request.session_id, message, turn, 10)
            state = agent.get_state(request.session_id)
            wrapper_state["turn"] = turn
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive presentation boundary
        LOGGER.exception("Agent failed for session_id=%s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail="The Shopping Copilot could not process this message. Please try again.",
        ) from exc

    return {
        "turn": turn,
        "agent": {
            "message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "usage": response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
        },
        "recommendations": _enrich_recommendations(response.get("recommendations")),
        "state": state,
        "developer": {"agent_response": response, "state": state},
    }


@app.get("/api/eval/samples")
async def evaluator_samples() -> dict[str, Any]:
    return {"samples": scenario_runner.sample_list()}


@app.post("/api/eval/session")
async def create_evaluator_session(request: EvaluatorSessionRequest) -> dict[str, Any]:
    session_id = f"eval_{uuid.uuid4().hex}"
    try:
        with session_lock:
            sample = scenario_runner.samples_by_id.get(request.sample_id)
            if sample is None:
                raise KeyError(f"Public sample not found: {request.sample_id}")
            agent.reset(session_id, dict(sample.get("user_profile") or {}))
            sessions[session_id] = {"turn": 0}
            scenario = scenario_runner.start(session_id, request.sample_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Failed to create evaluator demo session")
        raise HTTPException(status_code=500, detail="Unable to start evaluator demo.") from exc
    return {"session_id": session_id, "turn": 0, "scenario": scenario}


@app.post("/api/eval/next")
async def run_evaluator_turn(request: EvaluatorActionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            wrapper_state = sessions.get(request.session_id)
            if wrapper_state is None:
                raise KeyError("Evaluator demo session not found")
            user_message = scenario_runner.current_message(request.session_id)
            turn = wrapper_state["turn"] + 1
            response = agent.respond(request.session_id, user_message, turn, 10)
            state = agent.get_state(request.session_id)
            wrapper_state["turn"] = turn
            scenario = scenario_runner.advance(request.session_id, response)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Evaluator demo turn failed")
        raise HTTPException(status_code=500, detail="Evaluator demo turn failed.") from exc
    return {
        "turn": turn,
        "user_message": user_message,
        "agent": {
            "message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "usage": response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
        },
        "recommendations": _enrich_recommendations(response.get("recommendations")),
        "state": state,
        "scenario": scenario,
        "developer": {"agent_response": response, "state": state},
    }


def _developer_error(exc: Exception, *, status_code: int = 400) -> HTTPException:
    LOGGER.exception("Developer trace operation failed")
    return HTTPException(status_code=status_code, detail=str(exc))


@app.post("/api/dev/session")
async def create_developer_session(request: SessionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            return trace_runner.create_session(request.user_profile)
    except Exception as exc:  # pragma: no cover - defensive presentation boundary
        raise _developer_error(exc, status_code=500) from exc


@app.post("/api/dev/scenario")
async def create_developer_scenario(request: EvaluatorSessionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            sample = scenario_runner.samples_by_id.get(request.sample_id)
            if sample is None:
                raise KeyError(f"Public sample not found: {request.sample_id}")
            session = trace_runner.create_session(dict(sample.get("user_profile") or {}))
            scenario = scenario_runner.start(session["session_id"], request.sample_id)
            return {**session, "scenario": scenario}
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except Exception as exc:
        raise _developer_error(exc, status_code=500) from exc


@app.post("/api/dev/turn")
async def start_developer_turn(request: DeveloperTurnRequest) -> dict[str, Any]:
    try:
        with session_lock:
            return _with_evaluation_preview(request.session_id, trace_runner.start_turn(
                request.session_id,
                request.message,
                top_k=request.top_k,
                retrieval_k=request.retrieval_k,
            ))
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except (RuntimeError, ValueError) as exc:
        raise _developer_error(exc, status_code=409) from exc


@app.post("/api/dev/stage")
async def run_developer_stage(request: DeveloperStageRequest) -> dict[str, Any]:
    try:
        with session_lock:
            return _with_evaluation_preview(request.session_id, trace_runner.run_stage(request.session_id, request.stage))
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except (RuntimeError, ValueError) as exc:
        raise _developer_error(exc, status_code=409) from exc
    except Exception as exc:  # stage error remains stored for later inspection
        LOGGER.exception("Developer stage %s failed", request.stage)
        with session_lock:
            trace = trace_runner.trace_view(request.session_id, selected_stage=request.stage)
        trace["operation_error"] = str(exc)
        _with_evaluation_preview(request.session_id, trace)
        return trace


@app.post("/api/dev/next")
async def run_developer_next(request: DeveloperActionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            return _with_evaluation_preview(request.session_id, trace_runner.run_next(request.session_id))
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except Exception as exc:
        LOGGER.exception("Developer Run Next failed")
        with session_lock:
            trace = trace_runner.trace_view(request.session_id)
        trace["operation_error"] = str(exc)
        return trace


@app.post("/api/dev/all")
async def run_developer_all(request: DeveloperActionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            return _with_evaluation_preview(request.session_id, trace_runner.run_all(request.session_id))
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except Exception as exc:
        LOGGER.exception("Developer Run All failed")
        with session_lock:
            trace = trace_runner.trace_view(request.session_id)
        trace["operation_error"] = str(exc)
        return trace


@app.post("/api/dev/restart")
async def restart_developer_turn(request: DeveloperActionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            return trace_runner.restart_turn(request.session_id)
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except (RuntimeError, ValueError) as exc:
        raise _developer_error(exc, status_code=409) from exc


@app.post("/api/dev/commit")
async def commit_developer_turn(request: DeveloperActionRequest) -> dict[str, Any]:
    try:
        with session_lock:
            developer_session = trace_runner.sessions.get(request.session_id)
            if developer_session is None or developer_session.active is None:
                raise KeyError("Developer session not found")
            was_committed = developer_session.active.committed
            response = developer_session.active.response
            result = trace_runner.commit_turn(request.session_id)
            if request.session_id in scenario_runner.sessions:
                if not was_committed:
                    if response is None:
                        raise RuntimeError("Final response is unavailable")
                    scenario = scenario_runner.advance(request.session_id, response)
                    if not scenario["done"] and scenario.get("next_user_message"):
                        trace_runner.start_turn(
                            request.session_id,
                            scenario["next_user_message"],
                            top_k=10,
                            retrieval_k=100,
                        )
                        result = trace_runner.session_view(request.session_id)
                else:
                    scenario = scenario_runner.view(request.session_id)
                result["scenario"] = scenario
            return result
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
    except RuntimeError as exc:
        raise _developer_error(exc, status_code=409) from exc


@app.get("/api/dev/trace/{session_id}/{turn}")
async def developer_history_trace(session_id: str, turn: int) -> dict[str, Any]:
    try:
        with session_lock:
            return trace_runner.history_trace(session_id, turn)
    except KeyError as exc:
        raise _developer_error(exc, status_code=404) from exc
