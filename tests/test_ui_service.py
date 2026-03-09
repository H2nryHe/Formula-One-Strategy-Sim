from tests.test_eval import _seed_eval_db

from f1sim.state import CarState, CleaningFlags, RaceState, TrackStatus, TyreCompound
from f1sim.ui.service import (
    build_demo_payload,
    build_key_moments,
    build_race_situation_panel,
    build_timeline_rows,
    list_drivers,
    list_sessions,
    max_lap_for_session,
)


def test_ui_service_lists_sessions_and_replays_payload(tmp_path) -> None:
    db_path = tmp_path / "ui.sqlite"
    _seed_eval_db(str(db_path))

    sessions = list_sessions(str(db_path))

    assert len(sessions) == 1
    assert sessions[0].session_id == "2023_unit_test_gp_r"
    assert max_lap_for_session(str(db_path), sessions[0].session_id) == 6
    assert list_drivers(str(db_path), sessions[0].session_id) == ("VER", "LEC")

    state, bundle = build_demo_payload(
        db_path=str(db_path),
        session_id=sessions[0].session_id,
        driver_id="VER",
        lap=4,
        seed=7,
        top_k=3,
        horizon_laps=6,
        n_scenarios=6,
    )

    assert state.session_id == sessions[0].session_id
    assert state.lap == 4
    assert bundle.session_id == sessions[0].session_id
    assert bundle.target_driver == "VER"
    assert bundle.top_k
    assert bundle.ground_truth["actual_action"] == "STAY_OUT"

    _, pit_bundle = build_demo_payload(
        db_path=str(db_path),
        session_id=sessions[0].session_id,
        driver_id="VER",
        lap=3,
        seed=7,
        top_k=3,
        horizon_laps=6,
        n_scenarios=6,
    )
    assert pit_bundle.ground_truth["actual_action"] == "PIT"


def test_ui_service_builds_timeline_rows_with_pit_markers(tmp_path) -> None:
    db_path = tmp_path / "ui.sqlite"
    _seed_eval_db(str(db_path))

    timeline_rows = build_timeline_rows(
        db_path=str(db_path),
        session_id="2023_unit_test_gp_r",
        driver_id="VER",
    )

    assert timeline_rows
    assert timeline_rows[0]["lap"] == 1
    assert "track_status" in timeline_rows[0]
    assert any(row["pit_in"] or row["pit_out"] for row in timeline_rows)


def test_ui_service_builds_situation_panel_and_key_moments(tmp_path) -> None:
    db_path = tmp_path / "ui.sqlite"
    _seed_eval_db(str(db_path))

    situation = build_race_situation_panel(
        db_path=str(db_path),
        session_id="2023_unit_test_gp_r",
        lap=4,
    )
    key_moments = build_key_moments(
        db_path=str(db_path),
        session_id="2023_unit_test_gp_r",
        driver_id="VER",
        seed=7,
        top_k=3,
        horizon_laps=6,
        n_scenarios=6,
    )

    assert "undercut_candidates" in situation
    assert "tyre_risk_ranking" in situation
    assert "traffic_hotspots" in situation
    assert isinstance(key_moments, tuple)


def test_situation_panel_filters_placeholder_position_cars(monkeypatch) -> None:
    state = RaceState(
        session_id="2023_test_r",
        lap=12,
        track_status=TrackStatus.GREEN,
        cars={
            "VER": CarState(
                driver_id="VER",
                team="Red Bull",
                position=1,
                gap_to_leader_ms=0.0,
                interval_ahead_ms=0.0,
                interval_behind_ms=1200.0,
                tyre_compound=TyreCompound.MEDIUM,
                tyre_age_laps=10,
                last_lap_time_ms=91000.0,
                cleaning_flags=CleaningFlags(is_traffic_heavy=False),
            ),
            "NOR": CarState(
                driver_id="NOR",
                team="McLaren",
                position=2,
                gap_to_leader_ms=1200.0,
                interval_ahead_ms=1200.0,
                interval_behind_ms=300.0,
                tyre_compound=TyreCompound.HARD,
                tyre_age_laps=14,
                last_lap_time_ms=91800.0,
                cleaning_flags=CleaningFlags(is_traffic_heavy=True),
            ),
            "TSU": CarState(
                driver_id="TSU",
                team="RB",
                position=999,
                gap_to_leader_ms=None,
                interval_ahead_ms=100.0,
                interval_behind_ms=100.0,
                tyre_compound=TyreCompound.SOFT,
                tyre_age_laps=20,
                last_lap_time_ms=None,
                cleaning_flags=CleaningFlags(is_traffic_heavy=True),
            ),
        },
    )

    monkeypatch.setattr(
        "f1sim.ui.service._session_states",
        lambda db_path, session_id: (state,),
    )

    situation = build_race_situation_panel(
        db_path="ignored.sqlite",
        session_id="2023_test_r",
        lap=12,
    )

    assert all(item["driver_id"] != "TSU" for item in situation["undercut_candidates"])
    assert all(item["driver_id"] != "TSU" for item in situation["traffic_hotspots"])
