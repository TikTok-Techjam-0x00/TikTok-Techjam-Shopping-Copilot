from __future__ import annotations

from pathlib import Path

from src.dialogue import decide_high_information_ask, record_asked_attribute
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
        self.retriever = Retriever.sota_default(str(self.catalog_path))
        self.reranker = QwenReranker(use_local_fallback=True)
        self.semantic_resolver = semantic_resolver
        self.semantic_policy = semantic_policy
        self._sessions: dict[str, ShoppingState] = {}
        self._last_asked: dict[str, str | None] = {}
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._recommended_asins: dict[str, set[str]] = {}
        self._recommendation_epoch: dict[str, int] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = create_state(session_id, user_profile)
        self._last_asked[session_id] = None
        self._conversations[session_id] = []
        self._recommended_asins[session_id] = set()
        self._recommendation_epoch[session_id] = 0

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
        # Once the first pool has been exhausted, inspect deeper rank windows
        # late in the conversation. Earlier turns retain the strongest page.
        if turn == 9:
            candidates_100 = self.retriever.retrieve_strata(
                query,
                state=state,
                intent=state.intent,
                windows=((0, 50), (400, 50)),
            )
        else:
            retrieval_page = {7: 1, 8: 2, 10: 3}.get(turn, 0)
            candidates_100 = self.retriever.retrieve_page(
                query,
                state=state,
                intent=state.intent,
                page=retrieval_page,
                page_size=100,
            )
        conversation_setter = getattr(self.reranker, "set_conversation", None)
        if callable(conversation_setter):
            conversation_setter(conversation)
        # A continued conversation is implicit negative feedback for products
        # already shown under the current intent. Keep the strongest ordering,
        # but avoid spending later recommendation slots on exact repeats. An
        # intent override starts a new constraint epoch and resets this memory.
        current_epoch = int(state.constraint_epoch)
        if self._recommendation_epoch.get(session_id) != current_epoch:
            self._recommended_asins[session_id].clear()
            self._recommendation_epoch[session_id] = current_epoch

        # A low-confidence early Top 10 can create an irreversible low-rank hit
        # before the customer's clarification arrives. During the first two
        # turns, expose only the strongest unseen candidate; a wrong Top 1 lets
        # the conversation continue and collect the missing requirements.
        recommendation_k = 1 if turn <= 2 else top_k
        local_fallback = getattr(self.reranker, "local_fallback", None)
        rank_all = getattr(local_fallback, "rank_all", None)
        previously_shown = self._recommended_asins[session_id]
        if callable(rank_all):
            ranked_all = rank_all(
                state,
                candidates_100,
            )
        else:
            requested_k = min(
                100,
                recommendation_k + len(previously_shown),
            )
            ranked_all = self.reranker.rerank(
                state,
                candidates_100,
                top_k=requested_k,
            )
        candidates_10 = [
            candidate
            for candidate in ranked_all
            if candidate.parent_asin not in previously_shown
        ][:recommendation_k]
        self._recommended_asins[session_id].update(
            candidate.parent_asin for candidate in candidates_10
        )
        decision = decide_high_information_ask(state, candidates_100)
        ask_attribute = decision["ask_attribute"]
        record_asked_attribute(state, ask_attribute)
        self._last_asked[session_id] = ask_attribute
        agent_message = decision["message"] or "Here are the closest matches I found."
        conversation.append({"role": "assistant", "content": agent_message})

        return {
            "message": agent_message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations_from_ranking(
                candidates_10,
                recommendation_k,
            ),
            "usage": {
                "prompt_tokens": int(getattr(self.reranker, "last_prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(self.reranker, "last_completion_tokens", 0) or 0),
            },
        }
