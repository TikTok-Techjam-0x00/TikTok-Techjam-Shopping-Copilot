from __future__ import annotations

import unittest

from fastapi import HTTPException

from frontend.server import (
    ChatRequest,
    DeveloperActionRequest,
    DeveloperStageRequest,
    DeveloperTurnRequest,
    EvaluatorActionRequest,
    EvaluatorSessionRequest,
    SessionRequest,
    chat,
    commit_developer_turn,
    create_developer_session,
    create_developer_scenario,
    create_evaluator_session,
    create_session,
    developer_history_trace,
    health,
    run_evaluator_turn,
    run_developer_all,
    run_developer_next,
    run_developer_stage,
    sessions,
    start_developer_turn,
)


class FrontendServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_health_reports_loaded_catalog(self) -> None:
        payload = health()
        self.assertEqual(payload["status"], "ok")
        self.assertGreater(payload["catalog_items"], 0)

    async def test_session_and_multi_turn_chat(self) -> None:
        created = await create_session(SessionRequest())
        session_id = created["session_id"]
        self.assertEqual(created["turn"], 0)

        first = await chat(ChatRequest(session_id=session_id, message="I'm looking for running shoes."))
        second = await chat(ChatRequest(session_id=session_id, message="Size 10, under $100."))

        self.assertEqual(first["turn"], 1)
        self.assertEqual(second["turn"], 2)
        self.assertEqual(sessions[session_id]["turn"], 2)
        self.assertEqual(second["state"]["session_id"], session_id)
        self.assertEqual(second["state"]["turn"], 2)
        self.assertIsInstance(first["agent"]["message"], str)
        self.assertLessEqual(len(first["recommendations"]), 10)
        for recommendation in first["recommendations"]:
            self.assertEqual(
                recommendation["parent_asin"],
                recommendation["product"]["parent_asin"],
            )

    async def test_invalid_session_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await chat(ChatRequest(session_id="missing", message="Find shoes"))
        self.assertEqual(context.exception.status_code, 404)

    async def test_turn_limit_is_enforced(self) -> None:
        created = await create_session(SessionRequest())
        sessions[created["session_id"]]["turn"] = 10
        with self.assertRaises(HTTPException) as context:
            await chat(ChatRequest(session_id=created["session_id"], message="One more"))
        self.assertEqual(context.exception.status_code, 409)

    async def test_developer_manual_pipeline_commit_and_turn_two(self) -> None:
        created = await create_developer_session(SessionRequest())
        session_id = created["session_id"]
        trace = await start_developer_turn(DeveloperTurnRequest(
            session_id=session_id,
            message="I need black running shoes under $100.",
        ))
        self.assertEqual(trace["stages"][0]["status"], "completed")
        self.assertTrue(all(stage["status"] == "not_run" for stage in trace["stages"][1:]))

        for stage_name in ("state", "query", "retrieval", "reranking", "dialogue", "response"):
            trace = await run_developer_stage(DeveloperStageRequest(
                session_id=session_id,
                stage=stage_name,
            ))
            self.assertEqual(trace["selected_stage"], stage_name)

        self.assertTrue(all(stage["status"] == "completed" for stage in trace["stages"]))
        retrieval = next(stage for stage in trace["stages"] if stage["name"] == "retrieval")
        self.assertEqual(len(retrieval["output"]), 100)
        duration = retrieval["duration_ms"]
        cached = await run_developer_stage(DeveloperStageRequest(
            session_id=session_id,
            stage="retrieval",
        ))
        self.assertEqual(
            next(stage for stage in cached["stages"] if stage["name"] == "retrieval")["duration_ms"],
            duration,
        )

        committed = await commit_developer_turn(DeveloperActionRequest(session_id=session_id))
        self.assertEqual(committed["turn"], 1)
        second = await start_developer_turn(DeveloperTurnRequest(
            session_id=session_id,
            message="Size 10.",
        ))
        self.assertIsNotNone(second["stages"][0]["input"]["previous_asked_attribute"])
        old = await developer_history_trace(session_id, 1)
        self.assertTrue(old["committed"])
        self.assertEqual(old["turn"], 1)

    async def test_developer_run_all_stores_every_stage(self) -> None:
        created = await create_developer_session(SessionRequest())
        session_id = created["session_id"]
        await start_developer_turn(DeveloperTurnRequest(
            session_id=session_id,
            message="Comfortable walking shoes.",
        ))
        trace = await run_developer_all(DeveloperActionRequest(session_id=session_id))
        self.assertEqual([stage["name"] for stage in trace["stages"]], [
            "input", "state", "query", "retrieval", "reranking", "dialogue", "response",
        ])
        self.assertTrue(all(stage["status"] == "completed" for stage in trace["stages"]))
        self.assertTrue(all(
            stage["duration_ms"] is not None
            for stage in trace["stages"][1:]
        ))

    async def test_developer_run_next_advances_one_stage_only(self) -> None:
        created = await create_developer_session(SessionRequest())
        session_id = created["session_id"]
        await start_developer_turn(DeveloperTurnRequest(
            session_id=session_id,
            message="A black jacket.",
        ))
        trace = await run_developer_next(DeveloperActionRequest(session_id=session_id))
        statuses = {stage["name"]: stage["status"] for stage in trace["stages"]}
        self.assertEqual(statuses["state"], "completed")
        self.assertEqual(statuses["query"], "not_run")

    async def test_evaluator_driven_demo_requires_no_manual_customer_reply(self) -> None:
        created = await create_evaluator_session(EvaluatorSessionRequest(sample_id="public_0001"))
        self.assertEqual(created["turn"], 0)
        self.assertTrue(created["scenario"]["next_user_message"])
        first = await run_evaluator_turn(EvaluatorActionRequest(session_id=created["session_id"]))
        self.assertEqual(first["turn"], 1)
        self.assertTrue(first["user_message"])
        self.assertIsInstance(first["agent"]["message"], str)
        if not first["scenario"]["done"]:
            self.assertTrue(first["scenario"]["next_user_message"])

    async def test_developer_scenario_prepares_next_message_after_commit(self) -> None:
        created = await create_developer_scenario(EvaluatorSessionRequest(sample_id="public_0002"))
        session_id = created["session_id"]
        first_message = created["scenario"]["next_user_message"]
        self.assertTrue(first_message)
        await start_developer_turn(DeveloperTurnRequest(
            session_id=session_id,
            message=first_message,
        ))
        completed = await run_developer_all(DeveloperActionRequest(session_id=session_id))
        self.assertIn(completed["evaluation_preview"]["status"], {"hit", "not_hit"})
        committed = await commit_developer_turn(DeveloperActionRequest(session_id=session_id))
        self.assertEqual(committed["turn"], 1)
        if not committed["scenario"]["done"]:
            self.assertTrue(committed["scenario"]["next_user_message"])
            self.assertEqual(committed["active_trace"]["turn"], 2)
            statuses = {stage["name"]: stage["status"] for stage in committed["active_trace"]["stages"]}
            self.assertEqual(statuses["input"], "completed")
            self.assertEqual(statuses["state"], "not_run")


if __name__ == "__main__":
    unittest.main()
