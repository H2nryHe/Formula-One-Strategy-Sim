"""Render a static HTML report from deterministic demo artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from f1sim.metrics import DELTA_TIME_DEFINITION_LABEL, DELTA_TIME_FORMULA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a simple HTML demo report from replay artifact files.",
    )
    parser.add_argument(
        "--artifact_dir",
        required=True,
        help=(
            "Directory containing config.json, eval_metrics.json, "
            "and recommendations jsonl files."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output HTML file.",
    )
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fmt_number(value: object, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value * 100:.1f}%"
    return "-"


def _render_kv_table(payload: dict[str, object]) -> str:
    rows: list[str] = []
    for key, value in payload.items():
        if isinstance(value, list):
            rendered = ", ".join(html.escape(str(item)) for item in value) or "-"
        else:
            rendered = html.escape(str(value))
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{rendered}</td>"
            "</tr>"
        )
    return "<table class='kv-table'>" + "".join(rows) + "</table>"


def _render_plan(plan: dict[str, object]) -> str:
    metrics = plan.get("metrics", {})
    diagnostics = plan.get("diagnostics", {})
    contributions = plan.get("contributions", {})
    actions = plan.get("actions", [])
    explanations = plan.get("explanations", [])
    counterfactuals = plan.get("counterfactuals", {})
    is_suspicious = bool(plan.get("is_suspicious"))
    suspicion_reason = plan.get("suspicion_reason")
    if actions:
        action_text = " -> ".join(
            f"{action.get('at_lap')}: {action.get('action')}" for action in actions
        )
    else:
        action_text = "STAY_OUT"

    explanation_blocks: list[str] = []
    for explanation in explanations:
        evidence = explanation.get("evidence", {})
        explanation_blocks.append(
            "<div class='explanation'>"
            f"<div class='reason'>{html.escape(str(explanation.get('code', '-')))}</div>"
            f"<p>{html.escape(str(explanation.get('text', '')))}</p>"
            f"{_render_kv_table(evidence if isinstance(evidence, dict) else {})}"
            "</div>"
        )

    cf_blocks: list[str] = []
    for label, payload in counterfactuals.items():
        if isinstance(payload, dict):
            cf_blocks.append(
                "<div class='counterfactual'>"
                f"<h5>{html.escape(str(label))}</h5>"
                f"{_render_kv_table(payload)}"
                "</div>"
            )

    metrics_html = (
        "<div class='metrics-grid'>"
        "<div><span>E[Δtime]</span><strong>"
        f"{_fmt_number(metrics.get('delta_time_mean_ms'))} ms"
        "</strong></div>"
        "<div><span>P10 / P50 / P90</span><strong>"
        f"{_fmt_number(metrics.get('delta_time_p10_ms'))} / "
        f"{_fmt_number(metrics.get('delta_time_p50_ms'))} / "
        f"{_fmt_number(metrics.get('delta_time_p90_ms'))}"
        "</strong></div>"
        "<div><span>Risk σ</span><strong>"
        f"{_fmt_number(metrics.get('risk_sigma_ms'))} ms"
        "</strong></div>"
        f"<div><span>P(gain≥1)</span><strong>{_fmt_pct(metrics.get('p_gain_pos_ge_1'))}</strong></div>"
        "</div>"
    )
    explanations_html = "".join(explanation_blocks)
    if not explanations_html:
        explanations_html = "<p class='muted'>No explanations</p>"
    counterfactuals_html = "".join(cf_blocks)
    if not counterfactuals_html:
        counterfactuals_html = "<p class='muted'>No counterfactuals</p>"
    warning_html = ""
    if is_suspicious:
        warning_text = html.escape(str(suspicion_reason or "No reason provided"))
        warning_html = (
            "<div class='warning-badge'>Suspicious Δtime</div>"
            f"<p class='warning-text'>{warning_text}</p>"
        )
    diagnostics_html = _render_kv_table(
        {
            "plan_total_time_ms": diagnostics.get("plan_total_time_ms", "-"),
            "baseline_total_time_ms": diagnostics.get("baseline_total_time_ms", "-"),
            "pit_loss_ms_used": diagnostics.get("pit_loss_ms_used", "-"),
            "delta_time_cap_ms": diagnostics.get("delta_time_cap_ms", "-"),
            "horizon_laps": diagnostics.get("horizon_laps", "-"),
            "n_scenarios": diagnostics.get("n_scenarios", "-"),
            "seed": diagnostics.get("seed", "-"),
        }
    )
    component_summary = diagnostics.get("per_lap_time_components_summary_ms", {})
    component_summary_html = _render_kv_table(
        component_summary if isinstance(component_summary, dict) else {}
    )
    contributions_html = _render_kv_table(
        contributions if isinstance(contributions, dict) else {}
    )

    return (
        "<div class='plan-card'>"
        f"<h4>{html.escape(str(plan.get('plan_id', '-')))}</h4>"
        f"<div class='muted'>Action: {html.escape(action_text)}</div>"
        f"{warning_html}"
        f"{metrics_html}"
        "<div class='subsection'><h5>Totals & Diagnostics</h5>"
        f"{diagnostics_html}"
        "<h5>Per-Lap Component Summary</h5>"
        f"{component_summary_html}"
        "</div>"
        "<div class='subsection'><h5>Contributions vs STAY_OUT</h5>"
        f"{contributions_html}"
        "</div>"
        "<div class='subsection'><h5>Explanations</h5>"
        f"{explanations_html}"
        + "</div>"
        "<div class='subsection'><h5>Counterfactuals</h5>"
        f"{counterfactuals_html}"
        + "</div>"
        "</div>"
    )


def _render_driver_section(driver_id: str, rows: list[dict[str, object]]) -> str:
    pit_rows: list[str] = []
    lap_blocks: list[str] = []
    empty_plans_html = "<p class='muted'>No plans</p>"

    for row in rows:
        lap = row.get("lap")
        bundle = row.get("recommendation_bundle", {})
        ground_truth = bundle.get("ground_truth", {})
        top_k = bundle.get("top_k", [])
        top_plan = top_k[0] if top_k else {}
        diagnostics = top_plan.get("diagnostics", {})
        predicted_action = diagnostics.get("immediate_action", "STAY_OUT")
        predicted_compound = (
            predicted_action.replace("PIT_TO_", "")
            if predicted_action.startswith("PIT_TO_")
            else "-"
        )
        actual_action = ground_truth.get("actual_action", "-")
        actual_compound = ground_truth.get("actual_compound_after") or "-"

        if actual_action == "PIT":
            pit_rows.append(
                "<tr>"
                f"<td>{lap}</td>"
                f"<td>{html.escape(str(predicted_action))}</td>"
                f"<td>{html.escape(str(actual_action))}</td>"
                f"<td>{html.escape(str(predicted_compound))}</td>"
                f"<td>{html.escape(str(actual_compound))}</td>"
                "</tr>"
            )

        pit_timeline = [
            (
                f"lap {item.get('lap')}: "
                f"{item.get('compound_before')} -> {item.get('compound_after')}"
            )
            for item in ground_truth.get("pit_timeline", [])
        ]
        ground_truth_html = _render_kv_table(
            {
                "actual_action": actual_action,
                "actual_compound_after": actual_compound,
                "pit_timeline": pit_timeline,
            }
        )
        plans_html = "".join(
            _render_plan(plan) for plan in top_k if isinstance(plan, dict)
        )
        if not plans_html:
            plans_html = empty_plans_html
        summary_text = (
            f"Lap {lap}: predicted {predicted_action} vs actual {actual_action}"
        )
        lap_blocks.append(
            "<details class='lap-block'>"
            f"<summary>{html.escape(summary_text)}</summary>"
            "<div class='lap-grid'>"
            "<div>"
            "<h4>Pred vs Actual</h4>"
            f"{ground_truth_html}"
            "</div>"
            "<div>"
            "<h4>Top-K Plans</h4>"
            f"{plans_html}"
            "</div>"
            "</div>"
            "</details>"
        )

    pit_table_html = (
        "<table class='data-table'>"
        "<thead><tr>"
        "<th>Lap</th>"
        "<th>Predicted Action</th>"
        "<th>Actual Action</th>"
        "<th>Predicted Compound</th>"
        "<th>Actual Compound</th>"
        "</tr></thead><tbody>"
        + "".join(pit_rows)
        + "</tbody></table>"
    )

    return (
        "<section class='driver-section'>"
        f"<h2>{html.escape(driver_id)}</h2>"
        "<div class='subsection'>"
        "<h3>Pred vs Actual on PIT Laps</h3>"
        + (
            pit_table_html
            if pit_rows
            else "<p class='muted'>No PIT laps in selected data.</p>"
        )
        + "</div>"
        "<div class='subsection'>"
        "<h3>Lap-by-Lap Top-K Explanations</h3>"
        + "".join(lap_blocks)
        + "</div>"
        "</section>"
    )


def render_html(artifact_dir: Path) -> str:
    config = _load_json(artifact_dir / "config.json")
    session_summary = _load_json(artifact_dir / "session_summary.json")
    metrics = _load_json(artifact_dir / "eval_metrics.json")

    rec_files = sorted(artifact_dir.glob("recommendations_driver_*.jsonl"))
    driver_sections = []
    for rec_file in rec_files:
        driver_id = rec_file.stem.split("_")[-1]
        rows = _load_jsonl(rec_file)
        driver_sections.append(_render_driver_section(driver_id, rows))

    behavioral = metrics.get("behavioral", {})
    decision_quality = metrics.get("decision_quality", {})
    stay_out = decision_quality.get("summary_by_baseline", {}).get("STAY_OUT", {})
    delta_time_config = config.get("delta_time", {})
    if not isinstance(delta_time_config, dict):
        delta_time_config = {}
    topk_overall = behavioral.get("topk_action_coverage", {}).get("overall", {})
    pm1_overall = behavioral.get("pit_window_hit_rate", {}).get("pm1", {}).get("overall", {})
    pm2_overall = behavioral.get("pit_window_hit_rate", {}).get("pm2", {}).get("overall", {})

    header_cards = [
        ("Session", config.get("session_id", "-")),
        ("Seed", config.get("seed", "-")),
        ("Top-1 Action Accuracy", _fmt_pct(behavioral.get("top1_action_accuracy"))),
        ("Pit Compound Accuracy", _fmt_pct(behavioral.get("pit_compound_accuracy"))),
        ("Rule Violation Rate", _fmt_number(decision_quality.get("rule_violation_rate"), 3)),
        ("STAY_OUT Mean Δtime", f"{_fmt_number(stay_out.get('mean'))} ms"),
    ]
    card_html = "".join(
        "<div class='stat-card'>"
        f"<div class='label'>{html.escape(str(label))}</div>"
        f"<div class='value'>{html.escape(str(value))}</div>"
        "</div>"
        for label, value in header_cards
    )
    metrics_rows = [
        (
            "Top-1 action accuracy",
            _fmt_number(behavioral.get("top1_action_accuracy"), 3),
            behavioral.get("top1_action_accuracy_n", "-"),
        ),
        (
            "Top-K coverage (K=3)",
            _fmt_number(topk_overall.get("value"), 3),
            topk_overall.get("n", "-"),
        ),
        (
            "Window hit ±1",
            _fmt_number(pm1_overall.get("value"), 3),
            pm1_overall.get("n", "-"),
        ),
        (
            "Window hit ±2",
            _fmt_number(pm2_overall.get("value"), 3),
            pm2_overall.get("n", "-"),
        ),
        (
            "Compound accuracy on PIT laps",
            _fmt_number(behavioral.get("pit_compound_accuracy"), 3),
            behavioral.get("pit_compound_accuracy_n", "-"),
        ),
        (
            "Rule violation rate",
            _fmt_number(decision_quality.get("rule_violation_rate"), 3),
            decision_quality.get("bundles_evaluated", "-"),
        ),
    ]
    metrics_table = (
        "<table class='data-table'>"
        "<thead><tr><th>Metric</th><th>Value</th><th>n</th></tr></thead><tbody>"
        + "".join(
            (
                f"<tr><td>{html.escape(str(name))}</td>"
                f"<td>{html.escape(str(value))}</td>"
                f"<td>{html.escape(str(count))}</td></tr>"
            )
            for name, value, count in metrics_rows
        )
        + "</tbody></table>"
    )

    session_table = _render_kv_table(
        {
            "lap_count": session_summary.get("lap_count"),
            "drivers": len(session_summary.get("drivers", [])),
            "pit_counts": ", ".join(
                f"{driver}: {count}"
                for driver, count in sorted(session_summary.get("pit_counts", {}).items())
            ),
            "track_status_summary": ", ".join(
                f"{status}: {count}"
                for status, count in sorted(
                    session_summary.get("track_status_summary", {}).items()
                )
            ),
        }
    )
    formula_text = html.escape(
        str(delta_time_config.get("formula", DELTA_TIME_FORMULA))
    )
    interpretation_text = html.escape(
        str(delta_time_config.get("interpretation", DELTA_TIME_DEFINITION_LABEL))
    )
    units_text = html.escape(str(delta_time_config.get("units", "ms")))
    assumptions_text = html.escape(str(config.get("assumptions_hash", "-")))
    model_versions_text = html.escape(
        json.dumps(config.get("model_versions", {}), sort_keys=True)
    )
    seed_text = html.escape(str(config.get("seed", "-")))
    horizon_text = html.escape(str(config.get("horizon_laps", "-")))
    top_k_text = html.escape(str(config.get("top_k", "-")))
    scenario_text = html.escape(str(config.get("n_scenarios", "-")))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>F1Sim Demo Report - Monza 2023</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: #fffdf9;
      --ink: #1f2430;
      --muted: #676d79;
      --line: #d8cfc1;
      --accent: #b33a3a;
      --accent-2: #203a5b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #efe7db 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2, h3, h4, h5 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.5; }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 24px;
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(32, 58, 91, 0.08);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .hero-meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 8px 16px;
      margin-top: 18px;
    }}
    .hero-meta p {{
      margin: 0;
    }}
    .stat-card, .plan-card, .subsection, .driver-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
    }}
    .stat-card {{
      padding: 14px 16px;
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 22px;
      font-weight: 700;
      color: var(--accent-2);
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
      margin-top: 18px;
    }}
    .definition-box {{
      margin-top: 18px;
      background: #f8efe4;
      border: 1px solid var(--line);
      border-left: 6px solid var(--accent);
      border-radius: 16px;
      padding: 18px 20px;
    }}
    .subsection {{
      padding: 18px;
    }}
    .kv-table, .data-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .kv-table th, .kv-table td, .data-table th, .data-table td {{
      text-align: left;
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      vertical-align: top;
    }}
    .kv-table th {{
      width: 220px;
      color: var(--muted);
      font-weight: 600;
    }}
    .data-table thead th {{
      border-top: 0;
      color: var(--accent-2);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .driver-section {{
      margin-top: 18px;
      padding: 18px;
    }}
    .lap-block {{
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffaf2;
      overflow: hidden;
    }}
    .lap-block summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 700;
      color: var(--accent-2);
    }}
    .lap-grid {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
      padding: 0 16px 16px;
    }}
    .plan-card {{
      padding: 16px;
      margin-top: 12px;
      background: #fff;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .metrics-grid span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metrics-grid strong {{
      display: block;
      margin-top: 4px;
      color: var(--accent);
      font-size: 16px;
    }}
    .reason {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #f2e1d4;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .warning-badge {{
      display: inline-block;
      margin-top: 10px;
      padding: 4px 9px;
      border-radius: 999px;
      background: #f9d9d0;
      color: #7f2016;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .warning-text {{
      color: #7f2016;
      margin: 10px 0 0;
    }}
    .explanation, .counterfactual {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .section-grid, .lap-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Monza 2023 Demo Report</h1>
      <p>
        Deterministic replay artifact view with direct Pred vs Actual comparison
        and full top-K explanation blocks.
      </p>
      <div class="stats">{card_html}</div>
      <div class="hero-meta">
        <p><strong>assumptions_hash:</strong> <code>{assumptions_text}</code></p>
        <p><strong>model_versions:</strong> <code>{model_versions_text}</code></p>
        <p><strong>seed:</strong> {seed_text}</p>
        <p><strong>horizon:</strong> {horizon_text}</p>
        <p><strong>top_k:</strong> {top_k_text}</p>
        <p><strong>n_scenarios:</strong> {scenario_text}</p>
      </div>
    </section>
    <section class="definition-box">
      <h2>Δtime Definition</h2>
      <p><strong>Formula:</strong> <code>{formula_text}</code></p>
      <p><strong>Interpretation:</strong> {interpretation_text}</p>
      <p><strong>Units:</strong> {units_text}</p>
    </section>
    <section class="section-grid">
      <div class="subsection">
        <h2>Run Config</h2>
        {_render_kv_table(config)}
      </div>
      <div class="subsection">
        <h2>Session Summary</h2>
        {session_table}
      </div>
    </section>
    <section class="subsection">
      <h2>Behavioral Summary</h2>
      {metrics_table}
    </section>
    {"".join(driver_sections)}
  </main>
</body>
</html>"""


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_path = Path(args.out)
    html_text = render_html(artifact_dir)
    output_path.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
