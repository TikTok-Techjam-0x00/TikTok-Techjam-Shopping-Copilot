"""Replay fixed pre-reranking cases across comparable ranking experiments."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item import Candidate, RankedCandidate
from src.retrieval import Catalog
from src.reranking import ConstraintMatcher, MatchStatus, SimpleReranker
from src.reranking.replay.provenance import collect_provenance, sha256_file
from src.reranking.replay.schema import SCHEMA_VERSION, ReplayCase, ReplayLabel


class FullRankingProtocol(Protocol):
    def rank_all(
        self,
        shopping_state: Mapping[str, Any],
        candidates_100: Sequence[Candidate],
    ) -> list[RankedCandidate]: ...


class RetrievalOrderRanker:
    """Control experiment: preserve the unique Retrieval order exactly."""

    def rank_all(
        self,
        shopping_state: Mapping[str, Any],
        candidates_100: Sequence[Candidate],
    ) -> list[RankedCandidate]:
        del shopping_state
        result: list[RankedCandidate] = []
        seen: set[str] = set()
        for candidate in candidates_100[:100]:
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
        return result


def builtin_experiments() -> dict[str, FullRankingProtocol]:
    return {
        "retrieval_order": RetrievalOrderRanker(),
        "s1_rule_fuzzy": SimpleReranker(),
    }


def _safe_run_id(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "-_.")
    if cleaned != value or not cleaned or cleaned in {".", ".."}:
        raise ValueError("run_id must contain only letters, digits, '-', '_' or '.'")
    return cleaned


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _experiment_config(ranker: FullRankingProtocol) -> dict[str, Any]:
    result: dict[str, Any] = {"class": f"{type(ranker).__module__}.{type(ranker).__qualname__}"}
    for name in ("strategy_config", "constraint_matcher", "feature_extractor", "relevance_scorer"):
        value = getattr(ranker, name, None)
        if value is None:
            continue
        if name in {"constraint_matcher", "feature_extractor", "relevance_scorer"}:
            nested: dict[str, Any] = {
                "class": f"{type(value).__module__}.{type(value).__qualname__}"
            }
            for config_name in ("config", "weights"):
                config = getattr(value, config_name, None)
                if config is not None:
                    nested[config_name] = _json_safe(config)
            result[name] = nested
        else:
            result[name] = _json_safe(value)
    return result


def _read_jsonl(path: Path, *, compressed: bool) -> list[dict[str, Any]]:
    opener = gzip.open if compressed else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"} if compressed else {"encoding": "utf-8"}
    with opener(path, **kwargs) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_replay_dataset(
    dataset_directory: str | Path,
) -> tuple[dict[str, Any], list[ReplayCase], dict[str, ReplayLabel]]:
    directory = Path(dataset_directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported replay schema: {manifest.get('schema_version')!r}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("manifest.files must be an object")

    def checked_file(key: str) -> Path:
        entry = files.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest.files.{key} must be an object")
        path = directory / str(entry.get("name") or "")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(entry.get("sha256") or ""):
            raise ValueError(f"replay {key} checksum mismatch")
        return path

    cases_path = checked_file("cases")
    labels_path = checked_file("labels")
    cases = [ReplayCase.from_dict(row) for row in _read_jsonl(cases_path, compressed=True)]
    labels_list = [ReplayLabel.from_dict(row) for row in _read_jsonl(labels_path, compressed=False)]
    labels = {label.case_id: label for label in labels_list}
    if len(labels) != len(labels_list):
        raise ValueError("duplicate case_id in replay labels")
    case_ids = {case.case_id for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("duplicate case_id in replay cases")
    if case_ids != set(labels):
        raise ValueError("replay cases and labels do not have identical case IDs")
    return manifest, cases, labels


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _restore_candidates(case: ReplayCase, catalog: Catalog) -> list[Candidate]:
    result: list[Candidate] = []
    for frozen in case.candidates_100:
        try:
            item = catalog[frozen.parent_asin]
        except KeyError as exc:
            raise ValueError(
                f"{case.case_id}: candidate {frozen.parent_asin!r} missing from catalog"
            ) from exc
        result.append(frozen.to_candidate(item))
    return result


def _hard_diagnostics(
    state: Mapping[str, Any],
    ranking: Sequence[RankedCandidate],
    matcher: ConstraintMatcher,
) -> dict[str, int]:
    hard = state.get("hard_constraint")
    hard_map = hard if isinstance(hard, Mapping) else {}
    comparison_count = 0
    violation_count = 0
    unknown_count = 0
    violating_items = 0
    for ranked in ranking[:10]:
        matches = matcher.match_candidate(ranked.item, hard=hard_map).hard
        comparison_count += len(matches)
        item_violated = False
        for match in matches:
            if match.status is MatchStatus.VIOLATED:
                violation_count += 1
                item_violated = True
            elif match.status is MatchStatus.UNKNOWN:
                unknown_count += 1
        violating_items += int(item_violated)
    return {
        "hard_comparison_count": comparison_count,
        "hard_violation_count": violation_count,
        "hard_unknown_count": unknown_count,
        "hard_violating_item_count": violating_items,
        "top10_item_count": min(10, len(ranking)),
    }


def _validate_ranking(case: ReplayCase, ranking: Sequence[RankedCandidate]) -> None:
    allowed = {candidate.parent_asin for candidate in case.candidates_100}
    returned = [candidate.parent_asin for candidate in ranking]
    if len(returned) != len(set(returned)):
        raise ValueError(f"{case.case_id}: ranker returned duplicate products")
    if not set(returned) <= allowed:
        raise ValueError(f"{case.case_id}: ranker returned products outside candidates_100")


def _case_result(
    case: ReplayCase,
    target: str,
    ranking: Sequence[RankedCandidate],
    latency_ms: float,
    matcher: ConstraintMatcher,
) -> dict[str, Any]:
    retrieval_ids = [candidate.parent_asin for candidate in case.candidates_100]
    reranked_ids = [candidate.parent_asin for candidate in ranking]
    retrieval_rank = retrieval_ids.index(target) + 1 if target in retrieval_ids else None
    rerank_rank = reranked_ids.index(target) + 1 if target in reranked_ids else None
    eligible = retrieval_rank is not None
    top10_hit = bool(case.scorable and rerank_rank is not None and rerank_rank <= 10)
    diagnostics = _hard_diagnostics(case.shopping_state, ranking, matcher)
    return {
        "case_id": case.case_id,
        "sample_id": case.sample_id,
        "scenario_type": case.scenario_type,
        "turn": case.turn,
        "scorable": case.scorable,
        "retrieval_covered": eligible,
        "retrieval_rank": retrieval_rank,
        "rerank_rank": rerank_rank,
        "top10_hit": top10_hit,
        "reciprocal_rank_at_10": 1.0 / rerank_rank if top10_hit and rerank_rank else 0.0,
        "promotion": bool(
            case.scorable
            and retrieval_rank is not None
            and retrieval_rank > 10
            and rerank_rank is not None
            and rerank_rank <= 10
        ),
        "demotion": bool(
            case.scorable
            and retrieval_rank is not None
            and retrieval_rank <= 10
            and (rerank_rank is None or rerank_rank > 10)
        ),
        "rank_change": (
            retrieval_rank - rerank_rank
            if case.scorable and retrieval_rank is not None and rerank_rank is not None
            else None
        ),
        "latency_ms": latency_ms,
        **diagnostics,
    }


def _session_metrics(case_results: Sequence[Mapping[str, Any]], max_turns: int) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_results:
        grouped[str(row["sample_id"])].append(row)
    sessions: list[dict[str, Any]] = []
    for sample_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["turn"]))
        first_hit = next((row for row in ordered if row["top10_hit"]), None)
        sessions.append(
            {
                "sample_id": sample_id,
                "scenario_type": str(ordered[0]["scenario_type"]),
                "hit": first_hit is not None,
                "first_hit_turn": int(first_hit["turn"]) if first_hit else None,
                "best_rank": int(first_hit["rerank_rank"]) if first_hit else None,
                "reciprocal_rank": float(first_hit["reciprocal_rank_at_10"]) if first_hit else 0.0,
            }
        )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
        hit_rate = sum(bool(row["hit"]) for row in rows) / len(rows)
        mrr = statistics.fmean(float(row["reciprocal_rank"]) for row in rows)
        mttc = statistics.fmean(
            int(row["first_hit_turn"]) if row["first_hit_turn"] is not None else max_turns + 1
            for row in rows
        )
        return {
            "sample_count": len(rows),
            "hit_rate_at_10": round(hit_rate, 6),
            "mrr": round(mrr, 6),
            "mttc": round(mttc, 6),
        }

    overall = summarize(sessions)
    efficiency = max(0.0, min(1.0, (max_turns + 1.0 - float(overall["mttc"] or max_turns + 1)) / max_turns))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    scenario_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for session in sessions:
        scenario_groups[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "replay_technical_score": round(technical_score, 6),
        "scenario_metrics": {
            scenario: summarize(rows) for scenario, rows in sorted(scenario_groups.items())
        },
        "sessions": sessions,
    }


def summarize_experiment(
    case_results: Sequence[Mapping[str, Any]],
    *,
    max_turns: int,
) -> dict[str, Any]:
    scorable = [row for row in case_results if row["scorable"]]
    eligible = [row for row in scorable if row["retrieval_covered"]]
    changes = [float(row["rank_change"]) for row in eligible if row["rank_change"] is not None]
    latencies = [float(row["latency_ms"]) for row in case_results]
    hard_comparisons = sum(int(row["hard_comparison_count"]) for row in scorable)
    hard_violations = sum(int(row["hard_violation_count"]) for row in scorable)
    hard_unknowns = sum(int(row["hard_unknown_count"]) for row in scorable)
    top10_items = sum(int(row["top10_item_count"]) for row in scorable)
    violating_items = sum(int(row["hard_violating_item_count"]) for row in scorable)
    case_count = len(scorable)
    eligible_count = len(eligible)
    return {
        "case_metrics": {
            "scorable_case_count": case_count,
            "retrieval_covered_count": eligible_count,
            "coverage_at_100": round(eligible_count / case_count, 6) if case_count else 0.0,
            "conditional_hit_at_10": round(
                sum(bool(row["top10_hit"]) for row in eligible) / eligible_count, 6
            ) if eligible_count else 0.0,
            "conditional_mrr_at_10": round(
                statistics.fmean(float(row["reciprocal_rank_at_10"]) for row in eligible), 6
            ) if eligible_count else 0.0,
            "promotion_count": sum(bool(row["promotion"]) for row in scorable),
            "demotion_count": sum(bool(row["demotion"]) for row in scorable),
            "mean_rank_change": round(statistics.fmean(changes), 6) if changes else None,
            "hard_violation_rate": round(hard_violations / hard_comparisons, 6)
            if hard_comparisons else 0.0,
            "hard_violation_item_rate_at_10": round(violating_items / top10_items, 6)
            if top10_items else 0.0,
            "hard_unknown_rate": round(hard_unknowns / hard_comparisons, 6)
            if hard_comparisons else 0.0,
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "session_metrics": _session_metrics(case_results, max_turns),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Reranking Replay Report",
        "",
        f"- Dataset run: `{report['dataset_run_id']}`",
        f"- Evaluation run: `{report['evaluation_run_id']}`",
        f"- Dataset Git commit: `{report['dataset_git_commit']}`",
        f"- Evaluation Git commit: `{report['evaluation_git_commit']}`",
        "",
        "## Aggregate comparison",
        "",
        "| Experiment | Cond. Hit@10 | Cond. MRR@10 | Promotions | Demotions | Mean rank Δ | Hard violation | P95 ms | Replay score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["experiments"].items():
        case = result["summary"]["case_metrics"]
        latency = result["summary"]["latency_ms"]
        session = result["summary"]["session_metrics"]
        rank_change = case["mean_rank_change"]
        lines.append(
            f"| {name} | {case['conditional_hit_at_10']:.6f} | "
            f"{case['conditional_mrr_at_10']:.6f} | {case['promotion_count']} | "
            f"{case['demotion_count']} | {rank_change if rank_change is not None else '—'} | "
            f"{case['hard_violation_item_rate_at_10']:.6f} | {latency['p95']:.3f} | "
            f"{session['replay_technical_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`coverage@100` is fixed by the recorded Retrieval output. Conditional metrics exclude cases where the target was not retrieved and Intent Override turns that were not yet scorable.",
            "",
            "The replay technical score is a counterfactual estimate valid while Dialogue consumes `candidates_100`, not reranked Top 10. Confirm selected changes with the official end-to-end evaluator.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_replay(
    dataset_directory: str | Path,
    *,
    catalog_path: str | Path,
    experiments: Mapping[str, FullRankingProtocol],
    output_root: str | Path | None = None,
    run_id: str | None = None,
    command: list[str] | None = None,
) -> Path:
    if not experiments:
        raise ValueError("at least one experiment is required")
    dataset_directory = Path(dataset_directory).resolve()
    catalog_path = Path(catalog_path).resolve()
    manifest, cases, labels = load_replay_dataset(dataset_directory)
    recorded_catalog = manifest["generation_provenance"]["inputs"]["catalog"]
    actual_catalog_hash = sha256_file(catalog_path)
    if actual_catalog_hash != recorded_catalog["sha256"]:
        raise ValueError("catalog checksum differs from the catalog used to record this replay dataset")
    catalog = Catalog.load(catalog_path)
    matcher = ConstraintMatcher()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evaluation_provenance = collect_provenance(
        PROJECT_ROOT,
        catalog_path=catalog_path,
        dataset_path=dataset_directory / "cases.jsonl.gz",
        dataset_role="reranking_replay_cases",
        command=command,
    )
    short_commit = evaluation_provenance["git"]["short_commit"]
    resolved_run_id = _safe_run_id(run_id or f"{timestamp}_{short_commit}")
    output_directory = Path(output_root).resolve() / resolved_run_id if output_root else dataset_directory / "results" / resolved_run_id
    output_directory.mkdir(parents=True, exist_ok=False)

    experiment_reports: dict[str, Any] = {}
    detailed_rows: list[dict[str, Any]] = []
    max_turns = int(manifest["generation_policy"]["max_turns"])
    for name, ranker in experiments.items():
        rows: list[dict[str, Any]] = []
        for case in cases:
            candidates = _restore_candidates(case, catalog)
            started = time.perf_counter()
            ranking = ranker.rank_all(case.shopping_state, candidates)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _validate_ranking(case, ranking)
            row = _case_result(
                case,
                labels[case.case_id].target_parent_asin,
                ranking,
                elapsed_ms,
                matcher,
            )
            rows.append(row)
            detailed_rows.append({"experiment": name, **row})
        experiment_reports[name] = {
            "config": _experiment_config(ranker),
            "summary": summarize_experiment(rows, max_turns=max_turns),
        }

    dataset_git = manifest["generation_provenance"]["git"]
    report = {
        "schema_version": "reranking-replay-report-v1",
        "dataset_run_id": manifest["run_id"],
        "evaluation_run_id": resolved_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_git_commit": dataset_git["commit"],
        "dataset_git_dirty": dataset_git["dirty"],
        "evaluation_git_commit": evaluation_provenance["git"]["commit"],
        "evaluation_git_dirty": evaluation_provenance["git"]["dirty"],
        "dataset_manifest": str((dataset_directory / "manifest.json").resolve()),
        "evaluation_provenance": evaluation_provenance,
        "experiments": experiment_reports,
    }
    report_path = output_directory / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    details_path = output_directory / "case_results.jsonl.gz"
    with gzip.open(details_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in detailed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output_directory / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay fixed cases against one or more Rerankers.")
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.jsonl")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["retrieval_order", "s1_rule_fuzzy"],
        choices=sorted(builtin_experiments()),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available = builtin_experiments()
    output = evaluate_replay(
        args.dataset_directory,
        catalog_path=args.catalog,
        experiments={name: available[name] for name in args.experiments},
        output_root=args.output_root,
        run_id=args.run_id,
        command=[sys.executable, "-m", "src.reranking.replay.evaluator", *(argv or sys.argv[1:])],
    )
    print(json.dumps({"replay_report": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
