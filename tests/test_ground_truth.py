import sqlite3

from tests.test_eval import _seed_eval_db

from f1sim.ground_truth import (
    attach_ground_truth_to_bundle,
    extract_lap_actions,
    extract_pit_calls,
    load_team_calls,
    materialize_team_calls,
)
from f1sim.ingest.fastf1_connector import FastF1Connector
from f1sim.replaydb import replay_state_at_lap
from f1sim.strategy import RolloutSearchConfig, RolloutStrategySearcher, build_model_suite


def test_extract_pit_calls_uses_pit_in_lap_convention() -> None:
    laps = [
        {
            "session_id": "s1",
            "driver_id": "VER",
            "lap_number": 1,
            "pit_in": 0,
            "pit_out": 0,
            "tyre_compound": "SOFT",
            "track_status": "GREEN",
            "lap_end_time_ms": 90000.0,
        },
        {
            "session_id": "s1",
            "driver_id": "VER",
            "lap_number": 2,
            "pit_in": 1,
            "pit_out": 0,
            "tyre_compound": "SOFT",
            "track_status": "GREEN",
            "lap_end_time_ms": 181000.0,
        },
        {
            "session_id": "s1",
            "driver_id": "VER",
            "lap_number": 3,
            "pit_in": 0,
            "pit_out": 1,
            "tyre_compound": "MEDIUM",
            "track_status": "GREEN",
            "lap_end_time_ms": 283000.0,
        },
    ]

    pit_calls = extract_pit_calls(laps)
    action_labels = extract_lap_actions(laps)

    assert len(pit_calls) == 1
    assert pit_calls[0].pit_lap == 2
    assert pit_calls[0].compound_before == "SOFT"
    assert pit_calls[0].compound_after == "MEDIUM"
    assert action_labels[("VER", 2)].actual_action == "PIT"
    assert action_labels[("VER", 2)].actual_compound_after == "MEDIUM"
    assert action_labels[("VER", 1)].actual_pit_window_label["w3"] == 1


def test_recommend_bundle_includes_ground_truth_fields(tmp_path) -> None:
    db_path = tmp_path / "ground_truth.sqlite"
    _seed_eval_db(str(db_path))
    materialize_team_calls(db_path=str(db_path), session_id="2023_unit_test_gp_r")
    pit_calls = load_team_calls(db_path=str(db_path), session_id="2023_unit_test_gp_r")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        laps = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM laps WHERE session_id = ?",
                ("2023_unit_test_gp_r",),
            )
        ]
    action_labels = extract_lap_actions(laps)

    state = replay_state_at_lap(
        db_path=str(db_path),
        session_id="2023_unit_test_gp_r",
        lap=3,
    )
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(horizon_laps=5, n_scenarios=6, top_k=3),
    )
    bundle = searcher.recommend(state, "VER", horizon_laps=5, top_k=3, seed=11)
    attach_ground_truth_to_bundle(bundle=bundle, action_labels=action_labels, pit_calls=pit_calls)

    assert bundle.ground_truth["actual_action"] == "PIT"
    assert bundle.ground_truth["actual_compound_after"] == "HARD"
    assert bundle.ground_truth["pit_timeline"]


def test_schema_includes_team_calls_table(tmp_path) -> None:
    db_path = tmp_path / "schema.sqlite"
    with sqlite3.connect(db_path) as conn:
        FastF1Connector.create_schema(conn)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert "team_calls" in tables
