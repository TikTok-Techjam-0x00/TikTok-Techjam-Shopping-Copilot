"""Record deterministic evaluator turns as versioned reranking replay data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from src.item import Candidate, RankedCandidate
from src.pipeline import Pipeline
from src.retrieval import Catalog
from src.reranking.replay.provenance import collect_provenance, sha256_file
from src.reranking.replay.schema import (
    SCHEMA_VERSION,
    ReplayCandidate,
    ReplayCase,
    ReplayLabel,
)


class RerankerProtocol(Protocol):
    def rerank(
        self,
        shopping_state: object,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> list[RankedCandidate]: ...


class ReplayRecorder:
    """Receive the exact inputs seen by a wrapped runtime Reranker."""

    def __init__(self) -> None:
        self.cases: list[ReplayCase] = []
        self.labels: list[ReplayLabel] = []
        self._context: dict[str, Any] | None = None

    def begin_case(
        self,
        *,
        sample_id: str,
        scenario_type: str,
        turn: int,
        target_parent_asin: str,
        scorable: bool,
        override_applied: bool,
    ) -> None:
        if self._context is not None:
            raise RuntimeError("previous replay case was not recorded")
        self._context = {
            "case_id": f"{sample_id}_turn_{turn:02d}",
            "sample_id": sample_id,
            "scenario_type": scenario_type,
            "turn": turn,
            "target_parent_asin": target_parent_asin,
            "scorable": scorable,
            "override_applied": override_applied,
        }

    def record(
        self,
        shopping_state: object,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
    ) -> None:
        context = self._context
        if context is None:
            raise RuntimeError("begin_case must be called before recording")
        if not hasattr(shopping_state, "to_dict"):
            raise TypeError("recorded shopping_state must provide to_dict()")
        compact_candidates: list[ReplayCandidate] = []
        seen: set[str] = set()
        for raw in candidates_100[:100]:
            candidate = raw if isinstance(raw, Candidate) else Candidate.from_dict(raw)
            if candidate.parent_asin in seen:
                continue
            seen.add(candidate.parent_asin)
            compact_candidates.append(ReplayCandidate.from_candidate(candidate))
        case = ReplayCase(
            case_id=context["case_id"],
            sample_id=context["sample_id"],
            scenario_type=context["scenario_type"],
            turn=context["turn"],
            scorable=context["scorable"],
            override_applied=context["override_applied"],
            shopping_state=dict(shopping_state.to_dict()),
            candidates_100=tuple(compact_candidates),
        )
        self.cases.append(case)
        self.labels.append(
            ReplayLabel(
                case_id=case.case_id,
                target_parent_asin=context["target_parent_asin"],
            )
        )
        self._context = None

    def assert_idle(self) -> None:
        if self._context is not None:
            raise RuntimeError("pipeline did not invoke the recording Reranker")


class RecordingReranker:
    """Transparent wrapper that records inputs and delegates runtime output."""

    def __init__(
        self,
        delegate: RerankerProtocol,
        recorder: ReplayRecorder,
        *,
        execute_delegate: bool = False,
    ) -> None:
        self.delegate = delegate
        self.recorder = recorder
        self.execute_delegate = execute_delegate

    def rerank(
        self,
        shopping_state: object,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
        top_k: int = 10,
    ) -> list[RankedCandidate]:
        self.recorder.record(shopping_state, candidates_100)
        if self.execute_delegate:
            return self.delegate.rerank(shopping_state, candidates_100, top_k=top_k)

        # The current Dialogue policy reads candidates_100 directly, and the
        # recorder deliberately ignores target-hit stopping. Recommendations do
        # not affect the next simulated message, so a retrieval-order passthrough
        # avoids paying for a Reranker whose output is discarded during capture.
        result: list[RankedCandidate] = []
        seen: set[str] = set()
        for raw in candidates_100[:100]:
            candidate = raw if isinstance(raw, Candidate) else Candidate.from_dict(raw)
            if candidate.parent_asin in seen:
                continue
            seen.add(candidate.parent_asin)
            result.append(
                RankedCandidate.from_candidate(
                    candidate,
                    rerank_rank=len(result) + 1,
                    rerank_score=float(candidate.retrieval_score or 0.0),
                    matched=[],
                    violation=[],
                )
            )
            if len(result) >= top_k:
                break
        return result


def _safe_run_id(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "-_.")
    if cleaned != value or not cleaned or cleaned in {".", ".."}:
        raise ValueError("run_id must contain only letters, digits, '-', '_' or '.'")
    return cleaned


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, compressed: bool) -> None:
    opener = gzip.open if compressed else Path.open
    kwargs = {"mode": "wt", "encoding": "utf-8", "newline": "\n"} if compressed else {
        "mode": "w", "encoding": "utf-8", "newline": "\n"
    }
    with opener(path, **kwargs) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _default_run_id(provenance: Mapping[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_data = provenance.get("git")
    short_commit = git_data.get("short_commit", "unknown") if isinstance(git_data, Mapping) else "unknown"
    suffix = "-dirty" if isinstance(git_data, Mapping) and git_data.get("source_dirty") else ""
    return f"{timestamp}_{short_commit}{suffix}"


def _catalog_from_pipeline(pipeline: Pipeline) -> Catalog:
    strategy = pipeline.retriever.strategy
    catalog = getattr(strategy, "catalog", None)
    if not isinstance(catalog, Catalog):
        raise TypeError("Replay recorder requires a Retriever backed by Catalog")
    return catalog


def collect_replay_dataset(
    *,
    catalog_path: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
    limit: int | None = None,
    max_turns: int = MAX_TURNS,
    execute_runtime_reranker: bool = False,
    progress_every: int = 0,
    command: list[str] | None = None,
) -> Path:
    """Record every deterministic turn, continuing after hits to avoid survival bias."""

    catalog_path = Path(catalog_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    output_root = Path(output_root).resolve()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS}")
    if progress_every < 0:
        raise ValueError("progress_every must not be negative")

    provenance = collect_provenance(
        PROJECT_ROOT,
        catalog_path=catalog_path,
        dataset_path=dataset_path,
        dataset_role="public_evaluator_set",
        command=command,
    )
    resolved_run_id = _safe_run_id(run_id or _default_run_id(provenance))
    run_directory = output_root / resolved_run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    samples = load_jsonl(dataset_path)
    if limit is not None:
        samples = samples[:limit]

    pipeline = Pipeline(catalog_path)
    catalog = _catalog_from_pipeline(pipeline)
    recorder = ReplayRecorder()
    pipeline.reranker = RecordingReranker(
        pipeline.reranker,
        recorder,
        execute_delegate=execute_runtime_reranker,
    )

    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id") or f"sample_{index:04d}")
        scenario = str(sample.get("scenario_type") or "unknown")
        ground_truth = sample.get("ground_truth")
        if not isinstance(ground_truth, Mapping):
            raise ValueError(f"{sample_id}: ground_truth must be an object")
        target = str(ground_truth.get("parent_asin") or "")
        if target not in catalog:
            raise ValueError(f"{sample_id}: target {target!r} is missing from catalog")
        card, behavior = materialize_hidden_fields(sample, {target: catalog[target].to_dict()})
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}

        session_id = f"replay::{sample_id}"
        user_profile = sample.get("user_profile")
        pipeline.reset(session_id, dict(user_profile) if isinstance(user_profile, Mapping) else {})
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(list(catalog[target].categories)),
            disclosed,
        )

        for turn in range(1, max_turns + 1):
            recorder.begin_case(
                sample_id=sample_id,
                scenario_type=scenario,
                turn=turn,
                target_parent_asin=target,
                scorable=override_applied,
                override_applied=override_applied,
            )
            response = pipeline.respond(session_id, user_message, turn, TOP_K)
            recorder.assert_idle()
            if turn == max_turns:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        if progress_every and (index % progress_every == 0 or index == len(samples)):
            print(
                json.dumps(
                    {
                        "recorded_samples": index,
                        "total_samples": len(samples),
                        "recorded_cases": len(recorder.cases),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    cases_path = run_directory / "cases.jsonl.gz"
    labels_path = run_directory / "labels.jsonl"
    _write_jsonl(cases_path, [case.to_dict() for case in recorder.cases], compressed=True)
    _write_jsonl(labels_path, [label.to_dict() for label in recorder.labels], compressed=False)

    scenario_counts = Counter(case.scenario_type for case in recorder.cases)
    scorable_count = sum(case.scorable for case in recorder.cases)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen pre-reranking inputs for offline Module 3A comparison",
        "generation_policy": {
            "continue_after_target_hit": True,
            "reason": "Avoid baseline-dependent survival bias; Dialogue currently reads candidates_100.",
            "target_label_visible_to_reranker": False,
            "top_k": TOP_K,
            "max_turns": max_turns,
            "retrieval_k": 100,
            "runtime_reranker_execution": (
                "delegate" if execute_runtime_reranker else "retrieval_order_passthrough"
            ),
            "runtime_reranker_output_affects_dialogue": False,
        },
        "counts": {
            "samples": len(samples),
            "cases": len(recorder.cases),
            "scorable_cases": scorable_count,
            "labels": len(recorder.labels),
            "scenario_cases": dict(sorted(scenario_counts.items())),
        },
        "files": {
            "cases": {"name": cases_path.name, "sha256": sha256_file(cases_path)},
            "labels": {"name": labels_path.name, "sha256": sha256_file(labels_path)},
        },
        "generation_provenance": provenance,
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record versioned reranking replay cases.")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "public_set.jsonl")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reranking_replay",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument(
        "--execute-runtime-reranker",
        action="store_true",
        help="Also execute the current Reranker during recording (slower; not needed by current Dialogue).",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_directory = collect_replay_dataset(
        catalog_path=args.catalog,
        dataset_path=args.dataset,
        output_root=args.output_root,
        run_id=args.run_id,
        limit=args.limit,
        max_turns=args.max_turns,
        execute_runtime_reranker=args.execute_runtime_reranker,
        progress_every=args.progress_every,
        command=[sys.executable, "-m", "src.reranking.replay.recorder", *(argv or sys.argv[1:])],
    )
    print(json.dumps({"replay_dataset": str(run_directory)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
