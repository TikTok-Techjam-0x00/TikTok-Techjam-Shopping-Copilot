"""Render Retrieval experiment JSON as a dependency-free HTML report."""

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
body{font-family:Inter,Segoe UI,sans-serif;margin:0;background:#f5f7fb;color:#172033}
main{max-width:1180px;margin:0 auto;padding:32px}.card{background:white;border-radius:14px;
box-shadow:0 4px 18px #1c274c12;padding:20px;margin:16px 0}h1,h2{margin:0 0 14px}
.meta{color:#667085}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.metric{background:#eef3ff;border-radius:10px;padding:14px}.metric strong{font-size:24px;display:block}
.chart-row{display:grid;grid-template-columns:64px 1fr 74px;gap:10px;align-items:center;margin:8px 0}
.track{height:17px;background:#edf0f5;border-radius:9px;overflow:hidden}.bar{height:100%;background:#5577ee}
.bar.cumulative{background:#20a37a}table{border-collapse:collapse;width:100%}th,td{text-align:left;
padding:9px;border-bottom:1px solid #e5e7eb}details pre{white-space:pre-wrap;overflow-wrap:anywhere}
"""


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_cards(values: Mapping[str, Any]) -> str:
    cards = []
    for key, value in values.items():
        if isinstance(value, (str, int, float)) or value is None:
            cards.append(
                f'<div class="metric"><span>{html.escape(str(key))}</span>'
                f'<strong>{html.escape(_number(value))}</strong></div>'
            )
    return '<div class="metrics">' + "".join(cards) + "</div>"


def _multiturn_charts(result: Mapping[str, Any]) -> str:
    sections: list[str] = []
    ks = [int(value) for value in result.get("ks", [])]
    turns = result.get("turn_metrics", [])
    if not isinstance(turns, Sequence):
        return ""
    for k in ks:
        rows: list[str] = []
        for entry in turns:
            if not isinstance(entry, Mapping):
                continue
            strict = float(entry.get("strict_recall", {}).get(f"recall_at_{k}", 0.0))
            cumulative = float(entry.get(f"session_hit_rate_at_{k}", 0.0))
            rows.append(
                f'<div class="chart-row"><b>Turn {entry.get("turn")}</b>'
                f'<div><div class="track"><div class="bar" style="width:{strict * 100:.2f}%"></div></div>'
                f'<div class="track" style="margin-top:3px"><div class="bar cumulative" '
                f'style="width:{cumulative * 100:.2f}%"></div></div></div>'
                f'<span>{strict:.1%}<br>{cumulative:.1%}</span></div>'
            )
        sections.append(
            f'<section class="card"><h2>Recall@{k} 按轮变化</h2>'
            '<p class="meta">蓝色：该轮严格 Recall；绿色：截至该轮 Session HitRate。</p>'
            + "".join(rows)
            + "</section>"
        )
    return "".join(sections)


def _method_table(result: Mapping[str, Any]) -> str:
    methods = result.get("methods")
    if not isinstance(methods, Mapping):
        return ""
    ks = [int(value) for value in result.get("ks", [10, 50, 100])]
    header = "".join(f"<th>Recall@{k}</th>" for k in ks)
    rows = []
    for name, payload in methods.items():
        overall = payload.get("overall", {}) if isinstance(payload, Mapping) else {}
        cells = "".join(
            f'<td>{float(overall.get(f"recall_at_{k}", 0.0)):.1%}</td>' for k in ks
        )
        rows.append(f"<tr><th>{html.escape(str(name))}</th>{cells}</tr>")
    return (
        '<section class="card"><h2>方法比较</h2><table><thead><tr><th>Method</th>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_html(result: Mapping[str, Any]) -> str:
    title = str(result.get("experiment") or "Retrieval experiment")
    raw = html.escape(json.dumps(result, ensure_ascii=False, indent=2))
    overall = result.get("overall", {})
    cards = _metric_cards(overall) if isinstance(overall, Mapping) else ""
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1><p class=\"meta\">Retrieval 实验结果可视化</p>"
        f'<section class="card"><h2>总体指标</h2>{cards}</section>'
        f"{_multiturn_charts(result)}{_method_table(result)}"
        f'<details class="card"><summary>完整 JSON</summary><pre>{raw}</pre></details>'
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
