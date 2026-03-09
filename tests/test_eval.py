from __future__ import annotations

import json
import sqlite3

from f1sim.eval.report import run_session_evaluation, write_evaluation_outputs
from f1sim.ingest.fastf1_connector import FastF1Connector


def _seed_eval_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        FastF1Connector.create_schema(conn)
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, year, gp, session_type, event_name, circuit_name,
                source, start_time_utc, lap_count, loaded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2023_unit_test_gp_r",
                2023,
                "Unit Test GP",
                "R",
                "Unit Test GP",
                "Synthetic Ring",
                "synthetic",
                "2023-07-02T13:00:00+00:00",
                6,
                "2023-07-02T13:00:00+00:00",
            ),
        )
        conn.executemany(
            "INSERT INTO cars (session_id, driver_id, team_name, car_number) VALUES (?, ?, ?, ?)",
            [
                ("2023_unit_test_gp_r", "VER", "Red Bull", "1"),
                ("2023_unit_test_gp_r", "LEC", "Ferrari", "16"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO laps (
                session_id, driver_id, lap_number, position, lap_time_ms,
                sector1_time_ms, sector2_time_ms, sector3_time_ms,
                gap_to_leader_ms, interval_ahead_ms, tyre_compound,
                tyre_age_laps, track_status, is_accurate, pit_in, pit_out, lap_end_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "2023_unit_test_gp_r",
                    "VER",
                    1,
                    1,
                    91000.0,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    "MEDIUM",
                    16,
                    "1",
                    1,
                    0,
                    0,
                    91000.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "LEC",
                    1,
                    2,
                    91600.0,
                    None,
                    None,
                    None,
                    600.0,
                    600.0,
                    "MEDIUM",
                    10,
                    "1",
                    1,
                    0,
                    0,
                    91600.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "VER",
                    2,
                    1,
                    91450.0,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    "MEDIUM",
                    17,
                    "1",
                    1,
                    0,
                    0,
                    182450.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "LEC",
                    2,
                    2,
                    91800.0,
                    None,
                    None,
                    None,
                    950.0,
                    950.0,
                    "MEDIUM",
                    11,
                    "1",
                    1,
                    0,
                    0,
                    183400.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "VER",
                    3,
                    2,
                    113500.0,
                    None,
                    None,
                    None,
                    22500.0,
                    22500.0,
                    "MEDIUM",
                    18,
                    "1",
                    1,
                    1,
                    0,
                    295950.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "LEC",
                    3,
                    1,
                    91750.0,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    "MEDIUM",
                    12,
                    "1",
                    1,
                    0,
                    0,
                    275150.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "VER",
                    4,
                    2,
                    102500.0,
                    None,
                    None,
                    None,
                    3200.0,
                    3200.0,
                    "HARD",
                    0,
                    "4",
                    1,
                    0,
                    1,
                    398450.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "LEC",
                    4,
                    1,
                    120000.0,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    "MEDIUM",
                    13,
                    "4",
                    1,
                    0,
                    0,
                    395150.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "VER",
                    5,
                    1,
                    90800.0,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    "HARD",
                    1,
                    "1",
                    1,
                    0,
                    0,
                    489250.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "LEC",
                    5,
                    2,
                    91950.0,
                    None,
                    None,
                    None,
                    1150.0,
                    1150.0,
                    "MEDIUM",
                    14,
                    "1",
                    1,
                    0,
                    0,
                    487100.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "VER",
                    6,
                    1,
                    90950.0,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    "HARD",
                    2,
                    "1",
                    1,
                    0,
                    0,
                    580200.0,
                ),
                (
                    "2023_unit_test_gp_r",
                    "LEC",
                    6,
                    2,
                    92500.0,
                    None,
                    None,
                    None,
                    1550.0,
                    1550.0,
                    "MEDIUM",
                    15,
                    "1",
                    1,
                    0,
                    0,
                    579600.0,
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO weather (
                session_id, weather_idx, sample_time_ms, air_temp_c, track_temp_c,
                humidity, pressure, rainfall, wind_speed_ms, wind_direction_deg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("2023_unit_test_gp_r", 0, 60000.0, 24.0, 33.0, 50.0, 1008.0, 0.0, 1.2, 180.0),
                ("2023_unit_test_gp_r", 1, 360000.0, 24.5, 34.0, 49.0, 1007.8, 0.0, 1.5, 190.0),
            ],
        )
        conn.commit()


def test_run_session_evaluation_returns_stable_schema(tmp_path) -> None:
    db_path = tmp_path / "eval.sqlite"
    _seed_eval_db(str(db_path))

    report = run_session_evaluation(
        session_id="2023_unit_test_gp_r",
        db_path=str(db_path),
        horizon_laps=5,
        copy_policy="nearest",
    )

    assert report.session_id == "2023_unit_test_gp_r"
    assert "w1" in report.behavioral["pit_in_window"]
    assert "w3" in report.behavioral["pit_in_window"]
    assert "w5" in report.behavioral["pit_in_window"]
    assert report.behavioral["pit_timing_error"]["count"] >= 1
    assert report.behavioral["top1_action_accuracy"] is not None
    assert report.behavioral["topk_action_coverage"]["overall"]["value"] is not None
    assert report.behavioral["pit_window_hit_rate"]["pm1"]["overall"]["n"] >= 1
    assert (
        report.behavioral["pit_window_hit_rate"]["pm2"]["overall"]["value"]
        >= report.behavioral["pit_window_hit_rate"]["pm1"]["overall"]["value"]
    )
    assert report.behavioral["pit_compound_accuracy"] is not None
    assert "STAY_OUT" in report.decision_quality["summary_by_baseline"]
    assert "RULE_TYRE_AGE" in report.decision_quality["summary_by_baseline"]
    assert "COPY_NEAREST" in report.decision_quality["summary_by_baseline"]
    assert report.decision_quality["rule_violation_rate"] == 0.0
    assert report.model_versions["pace"] == "pace_v0.1"
    assert report.model_versions["pit_policy"] == "pit_policy_v0.1"
    assert report.model_versions["scenario"] == "scenario_v0.1"
    assert report.config["delta_time"]["formula"] == "baseline_total_time_ms - plan_total_time_ms"
    assert report.bundles
    first_bundle = report.bundles[0]["recommendation_bundle"]
    assert first_bundle["top_k"]
    assert "baselines" in first_bundle
    assert "ground_truth" in first_bundle
    assert first_bundle["model_versions"]["search"] == "rollout_search_v0.2"
    assert "diagnostics" in first_bundle["top_k"][0]
    assert "contributions" in first_bundle["top_k"][0]
    assert "plan_total_time_ms" in first_bundle["top_k"][0]["diagnostics"]
    assert report.ground_truth_summary["pits_per_driver"]


def test_write_evaluation_outputs_creates_json_and_markdown(tmp_path) -> None:
    db_path = tmp_path / "eval.sqlite"
    out_dir = tmp_path / "out"
    _seed_eval_db(str(db_path))

    outputs = write_evaluation_outputs(
        session_id="2023_unit_test_gp_r",
        db_path=str(db_path),
        out_dir=str(out_dir),
        horizon_laps=5,
        copy_policy="leader",
    )

    json_payload = json.loads((out_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "summary.md").read_text(encoding="utf-8")

    assert outputs["json_path"].endswith("evaluation_report.json")
    assert outputs["markdown_path"].endswith("summary.md")
    assert outputs["eval_metrics_path"].endswith("eval_metrics.json")
    assert outputs["eval_report_path"].endswith("eval_report.md")
    assert json_payload["session_id"] == "2023_unit_test_gp_r"
    assert "decision_quality" in json_payload
    assert "ground_truth_summary" in json_payload
    assert json_payload["decision_quality"]["rule_violation_rate"] == 0.0
    assert json_payload["config"]["delta_time"]["units"] == "ms"
    assert "Layer 1 Behavioral" in markdown
    assert "## Δtime Definition" in markdown
    assert "Top-K coverage (K=3)" in markdown
    assert "Window hit ±1" in markdown
    assert "COPY_LEADER" in markdown
    assert "Rule violation rate=0.000" in markdown
    assert "Pred vs Actual Sample" in markdown
