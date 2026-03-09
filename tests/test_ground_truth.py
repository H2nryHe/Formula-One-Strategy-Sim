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
    assert action_labels[("VER", 3)].actual_action == "STAY_OUT"
    assert action_labels[("VER", 3)].actual_compound_after is None
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


def test_load_team_calls_backfills_missing_table_for_legacy_db(tmp_path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                year INTEGER,
                gp TEXT,
                session_type TEXT,
                event_name TEXT,
                circuit_name TEXT,
                source TEXT,
                start_time_utc TEXT,
                lap_count INTEGER,
                loaded_at_utc TEXT
            );
            CREATE TABLE laps (
                session_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                lap_number INTEGER NOT NULL,
                pit_in INTEGER NOT NULL,
                pit_out INTEGER NOT NULL,
                tyre_compound TEXT,
                track_status TEXT,
                lap_end_time_ms REAL,
                PRIMARY KEY (session_id, driver_id, lap_number)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, year, gp, session_type, event_name, circuit_name,
                source, start_time_utc, lap_count, loaded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_r",
                2023,
                "Legacy GP",
                "R",
                "Legacy GP",
                "Legacy Ring",
                "synthetic",
                None,
                3,
                "2023-01-01T00:00:00+00:00",
            ),
        )
        conn.executemany(
            """
            INSERT INTO laps (
                session_id, driver_id, lap_number, pit_in, pit_out, tyre_compound,
                track_status, lap_end_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("legacy_r", "VER", 1, 0, 0, "SOFT", "GREEN", 90000.0),
                ("legacy_r", "VER", 2, 1, 0, "SOFT", "GREEN", 181000.0),
                ("legacy_r", "VER", 3, 0, 1, "MEDIUM", "GREEN", 283000.0),
            ],
        )
        conn.commit()

    pit_calls = load_team_calls(db_path=str(db_path), session_id="legacy_r")

    assert len(pit_calls) == 1
    assert pit_calls[0].pit_lap == 2


def test_load_team_calls_rebuilds_stale_per_lap_rows(tmp_path) -> None:
    db_path = tmp_path / "stale.sqlite"
    _seed_eval_db(str(db_path))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS team_calls (
                session_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                lap INTEGER NOT NULL,
                actual_action TEXT NOT NULL,
                compound_before TEXT,
                compound_after TEXT,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, driver_id, lap)
            )
            """
        )
        conn.execute("DELETE FROM team_calls WHERE session_id = ?", ("2023_unit_test_gp_r",))
        conn.executemany(
            """
            INSERT INTO team_calls (
                session_id, driver_id, lap, actual_action, compound_before,
                compound_after, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("2023_unit_test_gp_r", "VER", 1, "PIT", "MEDIUM", "HARD", "{}"),
                ("2023_unit_test_gp_r", "VER", 2, "PIT", "MEDIUM", "HARD", "{}"),
                ("2023_unit_test_gp_r", "VER", 3, "PIT", "MEDIUM", "HARD", "{}"),
            ],
        )
        conn.commit()

    pit_calls = load_team_calls(db_path=str(db_path), session_id="2023_unit_test_gp_r")

    assert len(pit_calls) == 1
    assert pit_calls[0].pit_lap == 3
    assert pit_calls[0].compound_after == "HARD"
