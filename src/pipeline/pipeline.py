from __future__ import annotations

from pathlib import Path

from src.dialogue import decide_ask, record_asked_attribute
from src.reranking import recommendations_from_ranking
from src.reranking_plugins import QwenReranker
from src.retrieval import Retriever
from src.state import (
    ShoppingState,
    SemanticPolicy,
    SemanticResolver,
    create_state,
    retrieval_query,
    sanitize_retrieval_text,
    update_state,
)


class Pipeline:
    """Coordinate State, Retrieval, Reranking, and Dialogue."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        semantic_resolver: SemanticResolver | None = None,
        semantic_policy: SemanticPolicy | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = Retriever.bm25_intent_routed(str(self.catalog_path))
        self.reranker = QwenReranker()
        self.semantic_resolver = semantic_resolver
        self.semantic_policy = semantic_policy
        self._sessions: dict[str, ShoppingState] = {}
        self._last_asked: dict[str, str | None] = {}
        self._conversations: dict[str, list[dict[str, str]]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = create_state(session_id, user_profile)
        self._last_asked[session_id] = None
        self._conversations[session_id] = []

    def get_state(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id].to_dict()

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        conversation = self._conversations[session_id]
        conversation.append({"role": "user", "content": user_message})

        state = update_state(
            self._sessions[session_id],
            user_message,
            turn=turn,
            asked_attribute=self._last_asked.get(session_id),
            semantic_resolver=self.semantic_resolver,
            semantic_policy=self.semantic_policy,
        )
        query = retrieval_query(state) or sanitize_retrieval_text(user_message)
        candidates_100 = self.retriever.retrieve(
            query,
            state=state,
            intent=state.intent,
            k=100,
        )
        self.reranker.set_conversation(conversation)
        candidates_10 = self.reranker.rerank(state, candidates_100, top_k=top_k)
        decision = decide_ask(state, candidates_100)
        ask_attribute = decision["ask_attribute"]
        record_asked_attribute(state, ask_attribute)
        self._last_asked[session_id] = ask_attribute
        agent_message = decision["message"] or "Here are the closest matches I found."
        conversation.append({"role": "assistant", "content": agent_message})

        return {
            "message": agent_message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations_from_ranking(candidates_10, top_k),
            "usage": {
                "prompt_tokens": self.reranker.last_prompt_tokens,
                "completion_tokens": self.reranker.last_completion_tokens,
            },
        }
