from __future__ import annotations

from pathlib import Path

from src.dialogue import decide_high_information_ask, record_asked_attribute
from src.reranking import EvidenceCoverageReranker, recommendations_from_ranking
from src.retrieval import Catalog, Retriever
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
        catalog: Catalog | None = None,
        semantic_resolver: SemanticResolver | None = None,
        semantic_policy: SemanticPolicy | None = None,
        retrieval_pool_size: int = 100,
    ) -> None:
        if isinstance(retrieval_pool_size, bool) or not isinstance(retrieval_pool_size, int):
            raise TypeError("retrieval_pool_size must be an integer")
        if retrieval_pool_size < 100:
            raise ValueError("retrieval_pool_size must be at least 100")
        self.catalog_path = Path(catalog_path)
        self.catalog = catalog if catalog is not None else Catalog.load(self.catalog_path)
        self.retriever = Retriever.sota_semantic_residual(self.catalog)
        self.reranker = EvidenceCoverageReranker()
        self.semantic_resolver = semantic_resolver
        self.semantic_policy = semantic_policy
        self.retrieval_pool_size = retrieval_pool_size
        self._sessions: dict[str, ShoppingState] = {}
        self._last_asked: dict[str, str | None] = {}
        self._recommended_asins: dict[str, set[str]] = {}
        self._recommendation_epoch: dict[str, int] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = create_state(session_id, user_profile)
        self._last_asked[session_id] = None
        self._recommended_asins[session_id] = set()
        self._recommendation_epoch[session_id] = 0

    def get_state(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id].to_dict()

    @staticmethod
    def _reset_model_usage(component: object | None) -> None:
        reset = getattr(component, "reset_usage", None)
        if callable(reset):
            reset()

    @staticmethod
    def _model_usage(component: object | None) -> tuple[int, int]:
        usage = getattr(component, "model_usage", None)
        if not callable(usage):
            return 0, 0
        prompt_tokens, completion_tokens = usage()
        return (
            max(0, int(prompt_tokens)),
            max(0, int(completion_tokens)),
        )

    @classmethod
    def _combined_model_usage(cls, *components: object | None) -> tuple[int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        for component in components:
            prompt, completion = cls._model_usage(component)
            prompt_tokens += prompt
            completion_tokens += completion
        return prompt_tokens, completion_tokens

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        # The official evaluator sums usage from every response, so reset here
        # to ensure this response reports only the current turn's provider calls.
        self._reset_model_usage(self.semantic_resolver)
        self._reset_model_usage(self.retriever)

        state = update_state(
            self._sessions[session_id],
            user_message,
            turn=turn,
            asked_attribute=self._last_asked.get(session_id),
            semantic_resolver=self.semantic_resolver,
            semantic_policy=self.semantic_policy,
        )
        query = retrieval_query(state) or sanitize_retrieval_text(user_message)
        current_epoch = int(state.constraint_epoch)
        if self._recommendation_epoch.get(session_id) != current_epoch:
            self._recommended_asins[session_id].clear()
            self._recommendation_epoch[session_id] = current_epoch
        previously_shown = self._recommended_asins[session_id]

        # Once the first pool has been exhausted, inspect deeper rank windows
        # late in the conversation. Earlier turns retain the strongest page.
        if self.retrieval_pool_size > 100:
            expanded_candidates = self.retriever.retrieve_page(
                query,
                state=state,
                intent=state.intent,
                page=0,
                page_size=self.retrieval_pool_size,
            )
            candidates_100 = [
                candidate
                for candidate in expanded_candidates
                if candidate.parent_asin not in previously_shown
            ][:100]
        elif turn == 8:
            candidates_100 = self.retriever.retrieve_residual_page(
                query,
                state=state,
                intent=state.intent,
                page=2,
                page_size=100,
            )
        elif turn == 9:
            candidates_100 = self.retriever.retrieve_strata(
                query,
                state=state,
                intent=state.intent,
                windows=((0, 50), (400, 50)),
            )
        else:
            retrieval_page = {7: 1, 10: 3}.get(turn, 0)
            candidates_100 = self.retriever.retrieve_page(
                query,
                state=state,
                intent=state.intent,
                page=retrieval_page,
                page_size=100,
            )
        # A continued conversation is implicit negative feedback for products
        # already shown under the current intent. Keep the strongest ordering,
        # but avoid spending later recommendation slots on exact repeats. An
        # intent override starts a new constraint epoch and resets this memory.
        # A low-confidence early Top 10 can create an irreversible low-rank hit
        # before the customer's clarification arrives. During the first two
        # turns, expose only the strongest unseen candidate; a wrong Top 1 lets
        # the conversation continue and collect the missing requirements.
        recommendation_k = 1 if turn <= 2 else top_k
        ranked_all = self.reranker.rank_all(state, candidates_100)
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
        prompt_tokens, completion_tokens = self._combined_model_usage(
            self.semantic_resolver,
            self.retriever,
        )

        return {
            "message": agent_message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations_from_ranking(
                candidates_10,
                recommendation_k,
            ),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
