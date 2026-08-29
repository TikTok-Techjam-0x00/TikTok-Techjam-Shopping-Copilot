"""Evaluate Retrieval recall on every turn of the public conversations.

This is a Module 1 diagnostic, not a replacement for the official evaluator.
It mirrors the official customer simulator and the current State/Dialogue path,
but scores the target's rank in Retrieval Top-K before reranking.  Sessions keep
running after a hit so that every turn has a comparable retrieval record.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)

from src.dialogue import decide_ask, record_asked_attribute
from src.item import Candidate
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.catalog import Catalog
from src.retrieval.dense import DenseRetriever
from src.retrieval.embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    OpenAIEmbeddingEncoder,
    default_embedding_cache_dir,
)
from src.retrieval.evaluation.first_turn import (
    DEFAULT_KS,
    _recall_summary,
    _target_asin,
    _validated_ks,
)
from src.retrieval.hybrid import HybridConfig, HybridRetriever
from src.retrieval.retriever import RetrievalStrategy
from src.retrieval.routing import IntentRoutedRetriever, IntentRoutingConfig
from src.retrieval.text import DEFAULT_TEXT_VERSION, TEXT_CONFIGS
from src.state import create_state, retrieval_query, sanitize_retrieval_text, update_state


AskPolicy = Callable[[object, Sequence[Candidate]], Mapping[str, Any] | str | None]
METHODS = ("bm25", "bm25_routed", "dense", "hybrid_rrf", "hybrid_weighted")


def _target_rank(candidates: Sequence[Candidate], target: str, max_k: int) -> int | None:
    seen: set[str] = set()
    target_rank: int | None = None
    for rank, candidate in enumerate(candidates[:max_k], start=1):
        if not isinstance(candidate, Candidate):
            raise TypeError("retriever must return shared Candidate objects")
        if candidate.parent_asin in seen:
            raise ValueError("retriever returned duplicate parent_asin values")
        seen.add(candidate.parent_asin)
        if candidate.parent_asin == target:
            target_rank = rank
    return target_rank


def _ask_attribute(
    policy: AskPolicy,
    state: object,
    candidates: Sequence[Candidate],
) -> str | None:
    decision = policy(state, candidates)
    if isinstance(decision, Mapping):
        value = decision.get("ask_attribute")
    else:
        value = decision
    return str(value) if value not in (None, "") else None


def _candidate_result(candidate: Candidate) -> dict[str, Any]:
    return {
        "parent_asin": candidate.parent_asin,
        "title": candidate.item.title,
        "retrieval_rank": candidate.retrieval_rank,
        "retrieval_score": candidate.retrieval_score,
        "bm25_score": candidate.bm25_score,
        "dense_score": candidate.dense_score,
    }


def _turn_metrics(
    session_rows: Sequence[Mapping[str, Any]],
    ks: Sequence[int],
    max_turns: int,
) -> list[dict[str, Any]]:
    session_ids = {str(row["sample_id"]) for row in session_rows}
    metrics: list[dict[str, Any]] = []
    for turn in range(1, max_turns + 1):
        current = [row for row in session_rows if int(row["turn"]) == turn]
        scorable = [row for row in current if bool(row["scorable"])]
        active = [row for row in scorable if bool(row["officially_active"])]
        ranks = [row["target_rank"] for row in active]
        diagnostic_ranks = [row["target_rank"] for row in scorable]
        row: dict[str, Any] = {
            "turn": turn,
            "session_count": len(current),
            "scorable_count": len(scorable),
            "officially_active_count": len(active),
            "strict_recall": _recall_summary(ranks, ks),
            "diagnostic_recall": _recall_summary(diagnostic_ranks, ks),
        }
        for k in ks:
            hit_sessions = {
                str(item["sample_id"])
                for item in session_rows
                if int(item["turn"]) <= turn
                and bool(item["scorable"])
                and bool(item["officially_active"])
                and item["target_rank"] is not None
                and int(item["target_rank"]) <= k
            }
            row[f"cumulative_hits_at_{k}"] = len(hit_sessions)
            row[f"remaining_unhit_at_{k}"] = len(session_ids) - len(hit_sessions)
            row[f"session_hit_rate_at_{k}"] = (
                round(len(hit_sessions) / len(session_ids), 6) if session_ids else 0.0
            )
        metrics.append(row)
    return metrics


def _overall_metrics(
    sessions: Sequence[Mapping[str, Any]],
    ks: Sequence[int],
    max_turns: int,
    *,
    include_counterfactual: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(sessions)}
    for k in ks:
        first_turns = [
            next(
                (
                    int(row["turn"])
                    for row in session["turns"]
                    if row["scorable"]
                    and (include_counterfactual or row["officially_active"])
                    and row["target_rank"] is not None
                    and int(row["target_rank"]) <= k
                ),
                None,
            )
            for session in sessions
        ]
        hits = sum(turn is not None for turn in first_turns)
        result[f"session_hits_at_{k}"] = hits
        result[f"session_hit_rate_at_{k}"] = (
            round(hits / len(sessions), 6) if sessions else 0.0
        )
        result[f"mean_first_hit_turn_at_{k}"] = (
            round(statistics.fmean(turn for turn in first_turns if turn is not None), 6)
            if hits
            else None
        )
        result[f"mttc_at_{k}"] = (
            round(
                statistics.fmean(
                    turn if turn is not None else max_turns + 1 for turn in first_turns
                ),
                6,
            )
            if sessions
            else None
        )
    return result


def evaluate_multiturn_recall(
    retriever: RetrievalStrategy,
    catalog: Catalog,
    samples: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    max_turns: int = MAX_TURNS,
    result_top_n: int = 10,
    stop_k: int = 10,
    continue_after_hit: bool = False,
    ask_policy: AskPolicy = decide_ask,
) -> dict[str, Any]:
    """Run state-aware Retrieval on every simulated turn and record exact ranks."""
    recall_ks = _validated_ks(ks)
    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")
    if result_top_n < 0:
        raise ValueError("result_top_n must not be negative")
    if stop_k <= 0:
        raise ValueError("stop_k must be positive")
    max_k = max(recall_ks[-1], stop_k)
    sessions: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample {index} must be an object")
        sample_id = str(sample.get("sample_id") or index)
        scenario = str(sample.get("scenario_type") or "unknown")
        target = _target_asin(sample)
        product = catalog.get(target)
        if product is None:
            raise ValueError(f"target parent_asin {target!r} is missing from the catalog")
        card, behavior = materialize_hidden_fields(dict(sample), {target: product.to_dict()})
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        state = create_state(f"retrieval-eval::{sample_id}", dict(sample.get("user_profile") or {}))
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(product.categories),
            disclosed,
        )
        last_asked: str | None = None
        first_official_hit_turn: int | None = None
        turns: list[dict[str, Any]] = []

        for turn in range(1, max_turns + 1):
            state = update_state(
                state,
                user_message,
                turn=turn,
                asked_attribute=last_asked,
            )
            query = retrieval_query(state) or sanitize_retrieval_text(user_message)
            started = time.perf_counter()
            candidates = retriever.retrieve(query, state=state, intent=state.intent, k=max_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            rank = _target_rank(candidates, target, max_k)
            officially_active = first_official_hit_turn is None
            official_hit = bool(
                officially_active
                and override_applied
                and rank is not None
                and rank <= stop_k
            )
            ask_attribute = _ask_attribute(ask_policy, state, candidates)
            record_asked_attribute(state, ask_attribute)
            last_asked = ask_attribute
            row = {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "turn": turn,
                "scorable": override_applied,
                "officially_active": officially_active,
                "post_hit_counterfactual": not officially_active,
                "official_hit": official_hit,
                "user_message": user_message,
                "query": query,
                "intent": getattr(state.intent, "value", state.intent),
                "ask_attribute": ask_attribute,
                "target_parent_asin": target,
                "target_rank": rank if override_applied else None,
                "raw_target_rank": rank,
                "latency_ms": round(latency_ms, 3),
                "hits": {
                    f"hit_at_{k}": bool(override_applied and rank is not None and rank <= k)
                    for k in recall_ks
                },
                "top_results": [
                    _candidate_result(candidate)
                    for candidate in candidates[:result_top_n]
                ],
            }
            turns.append(row)
            flat_rows.append(row)
            if official_hit:
                first_official_hit_turn = turn
                if not continue_after_hit:
                    break
            if turn == max_turns:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    ask_attribute,
                    disclosed,
                    boundary_used,
                )

        sessions.append(
            {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "target_parent_asin": target,
                "first_official_hit_turn": first_official_hit_turn,
                "turns": turns,
            }
        )

    grouped_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped_sessions[str(session["scenario_type"])].append(session)
        grouped_rows[str(session["scenario_type"])].extend(session["turns"])

    latencies = [float(row["latency_ms"]) for row in flat_rows]
    return {
        "evaluation_mode": (
            "continued_multiturn_retrieval_diagnostic"
            if continue_after_hit
            else "official_stop_multiturn_retrieval"
        ),
        "metric_definitions": {
            "strict_recall": (
                "Exact target in that turn's final Top-K among officially active, "
                "scorable sessions. Pre-override and post-hit turns are excluded."
            ),
            "diagnostic_recall": (
                "Exact target in all scorable rows, including explicitly marked "
                "post-hit counterfactual rows when continuation is enabled."
            ),
            "session_hit_rate": (
                "Fraction of all sessions that have retrieved the exact target at least "
                "once by this turn; final candidate lists remain strict Top-K."
            ),
            "stopping": (
                f"The official session stops when the target first enters Top-{stop_k}; "
                "Intent Override can stop only after the override becomes active."
            ),
            "continuation": (
                "continue_after_hit=true keeps post-hit turns only for diagnostics and "
                "marks them post_hit_counterfactual=true."
            ),
            "dialogue_dependency": (
                "The current deterministic 3B ask policy selects the next simulated user reply; "
                "the target label is never passed to Retrieval or Dialogue."
            ),
        },
        "ks": list(recall_ks),
        "max_turns": max_turns,
        "official_stop_k": stop_k,
        "continue_after_hit": continue_after_hit,
        "result_top_n": result_top_n,
        "overall": _overall_metrics(sessions, recall_ks, max_turns),
        "diagnostic_overall": _overall_metrics(
            sessions,
            recall_ks,
            max_turns,
            include_counterfactual=True,
        ),
        "turn_metrics": _turn_metrics(flat_rows, recall_ks, max_turns),
        "scenario_metrics": {
            scenario: {
                "overall": _overall_metrics(values, recall_ks, max_turns),
                "diagnostic_overall": _overall_metrics(
                    values,
                    recall_ks,
                    max_turns,
                    include_counterfactual=True,
                ),
                "turn_metrics": _turn_metrics(grouped_rows[scenario], recall_ks, max_turns),
            }
            for scenario, values in sorted(grouped_sessions.items())
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "sessions": sessions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record strict Retrieval Recall@K for every public-set turn."
    )
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--method", choices=METHODS, default="bm25")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--result-top-n", type=int, default=10)
    parser.add_argument("--stop-k", type=int, default=10)
    parser.add_argument(
        "--continue-after-hit",
        action="store_true",
        help="Keep post-hit turns as counterfactual diagnostics instead of official stopping.",
    )
    parser.add_argument("--bm25-text-version", choices=tuple(TEXT_CONFIGS), default=DEFAULT_TEXT_VERSION)
    parser.add_argument(
        "--browsing-text-version",
        choices=tuple(TEXT_CONFIGS),
        default="title_category_v1",
    )
    parser.add_argument(
        "--browsing-max-turn",
        type=int,
        default=1,
        help="Use the Browsing strategy through this turn, then return to Buying strategy.",
    )
    parser.add_argument("--dense-text-version", choices=tuple(TEXT_CONFIGS), default=DEFAULT_TEXT_VERSION)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--source-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    catalog = Catalog.load(args.catalog)
    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    bm25: BM25Retriever | None = None
    browsing_bm25: BM25Retriever | None = None
    dense: DenseRetriever | None = None
    encoder: OpenAIEmbeddingEncoder | None = None
    if args.method == "bm25":
        strategy: RetrievalStrategy = BM25Retriever(
            catalog,
            text_version=args.bm25_text_version,
        )
        bm25 = strategy
    elif args.method == "bm25_routed":
        bm25 = BM25Retriever(catalog, text_version=args.bm25_text_version)
        browsing_bm25 = BM25Retriever(
            catalog,
            text_version=args.browsing_text_version,
        )
        strategy = IntentRoutedRetriever(
            bm25,
            browsing_bm25,
            config=IntentRoutingConfig(browsing_max_turn=args.browsing_max_turn),
        )
    else:
        encoder = OpenAIEmbeddingEncoder.from_env()
        cache_dir = args.cache_dir or PROJECT_ROOT / default_embedding_cache_dir(
            encoder.model,
            encoder.dimension,
            args.dense_text_version,
        )
        dense = DenseRetriever(
            catalog,
            encoder,
            cache_dir,
            text_version=args.dense_text_version,
            query_embedding_mode="query_instruction",
            query_instruction=args.query_instruction,
        )
        if args.method == "dense":
            strategy = dense
        else:
            bm25 = BM25Retriever(catalog, text_version=args.bm25_text_version)
            strategy = HybridRetriever(
                bm25,
                dense,
                config=HybridConfig(
                    method="rrf" if args.method == "hybrid_rrf" else "weighted",
                    source_k=args.source_k,
                    rank_constant=args.rrf_constant,
                    alpha=args.alpha,
                    fallback_to_bm25=False,
                ),
            )

    try:
        result = evaluate_multiturn_recall(
            strategy,
            catalog,
            samples,
            ks=args.ks,
            max_turns=args.max_turns,
            result_top_n=args.result_top_n,
            stop_k=args.stop_k,
            continue_after_hit=args.continue_after_hit,
        )
    finally:
        if dense is not None:
            dense.close()
        if bm25 is not None:
            bm25.close()
        if browsing_bm25 is not None:
            browsing_bm25.close()

    result = {
        "experiment": f"multiturn_{args.method}_v1",
        "method": args.method,
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "catalog_items": len(catalog),
        "catalog_stats": asdict(catalog.stats),
        "bm25_text_version": args.bm25_text_version,
        "browsing_text_version": (
            args.browsing_text_version if args.method == "bm25_routed" else None
        ),
        "browsing_max_turn": (
            args.browsing_max_turn if args.method == "bm25_routed" else None
        ),
        "dense_text_version": args.dense_text_version if encoder is not None else None,
        "embedding_model": encoder.model if encoder is not None else None,
        "query_instruction": args.query_instruction if encoder is not None else None,
        "source_k": args.source_k if args.method.startswith("hybrid") else None,
        "rrf_constant": args.rrf_constant if args.method == "hybrid_rrf" else None,
        "alpha": args.alpha if args.method == "hybrid_weighted" else None,
        **result,
    }
    output = args.output or (
        PROJECT_ROOT / "artifacts" / "retrieval" / f"multiturn_{args.method}_v1.json"
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Full per-session result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
