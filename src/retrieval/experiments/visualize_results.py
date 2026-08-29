"""Render Retrieval experiment JSON as a compact dependency-free HTML report."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STYLE = """
*{box-sizing:border-box}body{font-family:Inter,Segoe UI,sans-serif;margin:0;background:#f5f7fb;color:#172033}
main{max-width:1120px;margin:0 auto;padding:28px}.card{background:white;border-radius:12px;box-shadow:0 3px 14px #1c274c12;
padding:20px;margin:14px 0}h1,h2,h3{margin:0 0 12px}.meta{color:#667085}.metrics{display:grid;
grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.metric{background:#eef3ff;border-radius:9px;padding:13px}
.metric strong{font-size:23px;display:block;margin-top:4px}.turn-row{display:grid;grid-template-columns:62px 1fr 230px;
gap:10px;align-items:center;margin:10px 0}.track{display:flex;height:18px;background:#e9edf3;border-radius:9px;overflow:hidden}
.hit{height:100%;background:#20a37a}.remaining{height:100%;background:#e05b67}.numbers{font-variant-numeric:tabular-nums}
.legend{display:flex;gap:18px;flex-wrap:wrap}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dot.hit{background:#20a37a}.dot.remaining{background:#e05b67}table{border-collapse:collapse;width:100%}th,td{text-align:left;
padding:8px;border-bottom:1px solid #e5e7eb}details summary{cursor:pointer;font-weight:600}.workflow{display:grid;
grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px;align-items:stretch}.step{background:#f4f6fa;border-radius:8px;padding:11px}
.gate-pass{color:#087f5b;font-weight:700}.gate-fail{color:#c92a2a;font-weight:700}@media(max-width:720px){main{padding:16px}
.turn-row{grid-template-columns:52px 1fr}.turn-row .numbers{grid-column:2}.card{padding:16px}}
"""


SCENARIO_LABELS = {
    "buying": "Buying",
    "browsing": "Browsing",
    "boundary": "Boundary",
    "intent_override": "Intent Override",
}


def _number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _percent(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _summary_cards(result: Mapping[str, Any]) -> str:
    overall = result.get("overall", {})
    if not isinstance(overall, Mapping):
        return ""
    values = (
        ("Session 数", overall.get("sample_count")),
        ("最终 Retrieval Hit@100", _percent(overall.get("session_hit_rate_at_100"))),
        ("MTTC@100", _number(overall.get("mttc_at_100"))),
        ("最终推荐 Hit@10", _percent(overall.get("session_hit_rate_at_10"))),
    )
    cards = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for label, value in values
    )
    return f'<section class="card"><h2>总体结果</h2><div class="metrics">{cards}</div></section>'


def _hit100_curve(title: str, result: Mapping[str, Any]) -> str:
    overall = result.get("overall", {})
    turns = result.get("turn_metrics", [])
    if not isinstance(overall, Mapping) or not isinstance(turns, Sequence):
        return ""
    total = int(overall.get("sample_count", 0))
    rows: list[str] = []
    for entry in turns:
        if not isinstance(entry, Mapping):
            continue
        rate = float(entry.get("session_hit_rate_at_100", 0.0))
        hits = int(entry.get("cumulative_hits_at_100", round(rate * total)))
        remaining = int(entry.get("remaining_unhit_at_100", max(0, total - hits)))
        hit_width = max(0.0, min(100.0, rate * 100.0))
        remaining_width = max(0.0, 100.0 - hit_width)
        rows.append(
            f'<div class="turn-row"><b>第 {entry.get("turn")} 轮</b>'
            f'<div class="track" role="img" aria-label="累计命中 {hits}，剩余未命中 {remaining}">'
            f'<div class="hit" style="width:{hit_width:.2f}%"></div>'
            f'<div class="remaining" style="width:{remaining_width:.2f}%"></div></div>'
            f'<span class="numbers">命中 {hits}/{total}（{rate:.1%}） · 未命中 {remaining}</span></div>'
        )
    final_rate = float(overall.get("session_hit_rate_at_100", 0.0))
    return (
        f'<section class="card"><h2>{html.escape(title)}</h2>'
        f'<p class="meta">目标进入 Retrieval Top100 即命中；累计 Hit@100 只增不减，最终 {final_rate:.1%}。</p>'
        '<p class="legend"><span><i class="dot hit"></i>累计命中</span>'
        '<span><i class="dot remaining"></i>剩余未命中</span></p>'
        + "".join(rows)
        + "</section>"
    )


def _primary_curves(result: Mapping[str, Any]) -> str:
    sections = [_hit100_curve("总体：逐轮 Retrieval Hit@100", result)]
    scenarios = result.get("scenario_metrics", {})
    if not isinstance(scenarios, Mapping):
        return "".join(sections)
    for key in ("buying", "browsing", "boundary", "intent_override"):
        payload = scenarios.get(key)
        if isinstance(payload, Mapping):
            sections.append(_hit100_curve(SCENARIO_LABELS[key], payload))
    return "".join(sections)


def _conditional_hit_rate(result: Mapping[str, Any], k: int = 100) -> float:
    """Return the denominator-weighted hit rate over all officially active turns."""
    turns = result.get("turn_metrics", [])
    if not isinstance(turns, Sequence):
        return 0.0
    hits = 0
    denominator = 0
    for entry in turns:
        if not isinstance(entry, Mapping):
            continue
        strict = entry.get("strict_recall", {})
        if not isinstance(strict, Mapping):
            continue
        hits += int(strict.get(f"hits_at_{k}", 0))
        denominator += int(strict.get("sample_count", 0))
    return hits / denominator if denominator else 0.0


def _scenario_overview_table(result: Mapping[str, Any]) -> str:
    scenarios = result.get("scenario_metrics", {})
    if not isinstance(scenarios, Mapping):
        return ""
    rows: list[str] = []
    for key in ("browsing", "buying", "boundary", "intent_override"):
        payload = scenarios.get(key)
        if not isinstance(payload, Mapping):
            continue
        overall = payload.get("overall", {})
        if not isinstance(overall, Mapping):
            continue
        total = int(overall.get("sample_count", 0))
        hits = int(overall.get("session_hits_at_100", 0))
        rows.append(
            f"<tr><th>{SCENARIO_LABELS[key]}</th>"
            f'<td>{_conditional_hit_rate(payload):.1%}</td>'
            f'<td><b>{_percent(overall.get("session_hit_rate_at_100"))}</b></td>'
            f'<td>{hits} / {total}</td><td>{max(0, total - hits)}</td></tr>'
        )
    return (
        '<section class="card"><h2>四场景 Retrieval 整体效果</h2>'
        '<p class="meta">Conditional Hit@100：所有仍参与评分轮次中的Top100命中率；'
        "Session HitRate@100：整段对话至少一次进入Top100的比例。</p>"
        "<table><thead><tr><th>场景</th><th>单轮 Conditional Hit@100</th>"
        "<th>整段对话 Session HitRate@100</th><th>命中 / Session</th><th>最终未命中</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _diagnostic_table(title: str, result: Mapping[str, Any]) -> str:
    turns = result.get("turn_metrics", [])
    if not isinstance(turns, Sequence):
        return ""
    rows: list[str] = []
    for entry in turns:
        if not isinstance(entry, Mapping):
            continue
        strict = entry.get("strict_recall", {})
        if not isinstance(strict, Mapping):
            strict = {}
        rows.append(
            "<tr>"
            f'<td>{entry.get("turn")}</td>'
            f'<td>{entry.get("officially_active_count", 0)}</td>'
            f'<td>{strict.get("hits_at_10", 0)}</td>'
            f'<td>{_percent(strict.get("recall_at_10"))}</td>'
            f'<td>{_percent(strict.get("recall_at_50"))}</td>'
            f'<td>{_percent(strict.get("recall_at_100"))}</td>'
            f'<td>{entry.get("remaining_unhit_at_50", "—")}</td>'
            f'<td>{entry.get("remaining_unhit_at_100", "—")}</td>'
            "</tr>"
        )
    return (
        f'<h3>{html.escape(title)}</h3><div style="overflow-x:auto"><table><thead><tr>'
        "<th>轮次</th><th>严格分母</th><th>当轮新命中@10</th><th>严格R@10</th>"
        "<th>严格R@50</th><th>严格R@100</th><th>未命中@50</th><th>未命中@100</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _diagnostics(result: Mapping[str, Any]) -> str:
    sections = [_diagnostic_table("总体", result)]
    scenarios = result.get("scenario_metrics", {})
    if isinstance(scenarios, Mapping):
        for key in ("buying", "browsing", "boundary", "intent_override"):
            payload = scenarios.get(key)
            if isinstance(payload, Mapping):
                sections.append(_diagnostic_table(SCENARIO_LABELS[key], payload))
    return (
        '<details class="card"><summary>展开严格 Recall 与 Top50/100 诊断</summary>'
        '<p class="meta">严格 Recall 的分母只包含当轮仍可计分的 Session，用来分析该轮对剩余难例的效果。</p>'
        + "".join(sections)
        + "</details>"
    )


def _method_table(result: Mapping[str, Any]) -> str:
    methods = result.get("methods")
    if not isinstance(methods, Mapping):
        return ""
    rows = []
    for name, payload in methods.items():
        overall = payload.get("overall", {}) if isinstance(payload, Mapping) else {}
        rows.append(
            f"<tr><th>{html.escape(str(name))}</th>"
            f'<td>{_percent(overall.get("recall_at_10"))}</td>'
            f'<td>{_percent(overall.get("recall_at_50"))}</td>'
            f'<td>{_percent(overall.get("recall_at_100"))}</td></tr>'
        )
    return (
        '<section class="card"><h2>方法比较</h2><table><thead><tr><th>方法</th>'
        "<th>Recall@10</th><th>Recall@50</th><th>Recall@100</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def _routing_comparison(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison")
    if not isinstance(comparison, Mapping):
        return ""
    rows: list[str] = []
    scenario_comparison = comparison.get("scenarios", {})
    groups: dict[str, object] = {"总体": comparison.get("overall")}
    if isinstance(scenario_comparison, Mapping):
        for key in ("buying", "browsing", "boundary", "intent_override"):
            if key in scenario_comparison:
                groups[SCENARIO_LABELS[key]] = scenario_comparison[key]
    for name, payload in groups.items():
        if not isinstance(payload, Mapping):
            continue
        baseline = payload.get("baseline", {})
        routed = payload.get("routed", {})
        if not isinstance(baseline, Mapping) or not isinstance(routed, Mapping):
            continue
        rows.append(
            f"<tr><th>{html.escape(name)}</th>"
            f'<td>{_percent(baseline.get("session_hit_rate_at_10"))} → '
            f'{_percent(routed.get("session_hit_rate_at_10"))}</td>'
            f'<td>{_percent(baseline.get("session_hit_rate_at_100"))} → '
            f'{_percent(routed.get("session_hit_rate_at_100"))}</td>'
            f'<td>{_number(baseline.get("mttc_at_10"))} → '
            f'{_number(routed.get("mttc_at_10"))}</td></tr>'
        )
    gate = result.get("reliability_gate", {})
    passed = bool(gate.get("passed")) if isinstance(gate, Mapping) else False
    gate_class = "gate-pass" if passed else "gate-fail"
    return (
        '<section class="card"><h2>基线对照</h2>'
        f'<p class="{gate_class}">可靠性门槛：{"通过，可升级基线" if passed else "未通过，不升级"}</p>'
        "<table><thead><tr><th>场景</th><th>最终 Hit@10</th><th>覆盖@100</th><th>MTTC@10</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _workflow() -> str:
    steps = (
        "固定数据集、对话模拟和 stop@10",
        "运行当前生产基线",
        "只修改一个 Retrieval 变量",
        "运行 200 Session 四场景评估",
        "比较逐轮 Hit@100、MTTC@100、最终 Hit@10",
        "通过门槛后替换基线并记录",
    )
    return (
        '<section class="card"><h2>固定实验流程</h2><div class="workflow">'
        + "".join(
            f'<div class="step"><b>{index}</b><br>{html.escape(step)}</div>'
            for index, step in enumerate(steps, 1)
        )
        + '</div><p class="meta">升级条件：总体 Hit@10、Hit@100 不下降，MTTC@10 不升高；'
        "四个场景均不能损失 Hit@10 或 Hit@100；完整测试必须通过。</p></section>"
    )


def render_html(result: Mapping[str, Any]) -> str:
    title = str(result.get("experiment") or "Retrieval experiment")
    routed = result.get("routed")
    display = routed if isinstance(routed, Mapping) else result
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        '<p class="meta">Retrieval 主指标看逐轮累计 Session Hit@100；Hit@10 与严格 Recall 作为结果和诊断指标。</p>'
        f"{_summary_cards(display)}{_routing_comparison(result)}"
        f"{_scenario_overview_table(display)}"
        f"{_primary_curves(display)}{_diagnostics(display)}"
        f"{_method_table(result)}{_workflow()}"
        '<p class="meta">完整 Session 明细保留在同名 JSON 文件中。</p>'
        "</main></body></html>"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Retrieval JSON metrics as HTML.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise SystemExit("input JSON must contain one object")
    output = args.output or args.input.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(result), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
