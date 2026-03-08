from tests.test_eval import _seed_eval_db

from f1sim.ui.service import (
    build_demo_payload,
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
