from f1sim.features import FEATURE_SCHEMA_VERSION, build_driver_features
from f1sim.state import CarLapUpdate, LapEndTick, StateEngine, TrackStatus, TyreCompound


def test_state_engine_updates_pit_stint_and_tyre_age() -> None:
    engine = StateEngine(clean_lap_window=3, traffic_gap_threshold_ms=400.0)

    lap_10 = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=10,
            track_status=TrackStatus.GREEN,
            weather={"air_c": 24.0, "track_c": 32.0},
            car_updates=[
                CarLapUpdate(
                    driver_id="VER",
                    team="Red Bull",
                    position=1,
                    lap_time_ms=91000.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    tyre_age_laps=5,
                ),
                CarLapUpdate(
                    driver_id="LEC",
                    team="Ferrari",
                    position=2,
                    lap_time_ms=91600.0,
                    gap_to_leader_ms=600.0,
                    interval_ahead_ms=600.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    tyre_age_laps=5,
                ),
            ],
        )
    )

    assert lap_10.cars["VER"].stint_id == 0
    assert lap_10.cars["VER"].tyre_age_laps == 5
    assert lap_10.cars["VER"].recent_lap_times_ms == [91000.0]

    lap_11 = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=11,
            track_status=TrackStatus.GREEN,
            car_updates=[
                CarLapUpdate(
                    driver_id="VER",
                    team="Red Bull",
                    position=2,
                    lap_time_ms=112000.0,
                    gap_to_leader_ms=22000.0,
                    interval_ahead_ms=22000.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    pit_in=True,
                ),
                CarLapUpdate(
                    driver_id="LEC",
                    team="Ferrari",
                    position=1,
                    lap_time_ms=91400.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    tyre_age_laps=6,
                ),
            ],
        )
    )

    assert lap_11.cars["VER"].stint_id == 0
    assert lap_11.cars["VER"].tyre_age_laps == 6
    assert lap_11.cars["VER"].pit_in is True
    assert lap_11.cars["VER"].cleaning_flags.is_inlap is True
    assert lap_11.cars["VER"].recent_lap_times_ms == [91000.0]

    lap_12 = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=12,
            track_status=TrackStatus.GREEN,
            car_updates=[
                CarLapUpdate(
                    driver_id="VER",
                    team="Red Bull",
                    position=2,
                    lap_time_ms=102500.0,
                    gap_to_leader_ms=3500.0,
                    interval_ahead_ms=3500.0,
                    tyre_compound=TyreCompound.HARD,
                    pit_out=True,
                ),
                CarLapUpdate(
                    driver_id="LEC",
                    team="Ferrari",
                    position=1,
                    lap_time_ms=99000.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    tyre_age_laps=7,
                ),
            ],
        )
    )

    assert lap_12.cars["VER"].stint_id == 1
    assert lap_12.cars["VER"].tyre_age_laps == 0
    assert lap_12.cars["VER"].pit_out is True
    assert lap_12.cars["VER"].cleaning_flags.is_outlap is True
    assert lap_12.cars["VER"].recent_lap_times_ms == [91000.0]
    assert lap_12.cars["VER"].used_dry_compounds == {"MEDIUM", "HARD"}
    assert lap_12.cars["VER"].used_wet is False

    lap_13 = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=13,
            track_status=TrackStatus.GREEN,
            car_updates=[
                CarLapUpdate(
                    driver_id="VER",
                    team="Red Bull",
                    position=2,
                    lap_time_ms=92000.0,
                    gap_to_leader_ms=1700.0,
                    interval_ahead_ms=1700.0,
                    tyre_compound=TyreCompound.HARD,
                ),
                CarLapUpdate(
                    driver_id="LEC",
                    team="Ferrari",
                    position=1,
                    lap_time_ms=91800.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    tyre_age_laps=8,
                ),
            ],
        )
    )

    assert lap_13.cars["VER"].stint_id == 1
    assert lap_13.cars["VER"].tyre_age_laps == 1
    assert lap_13.cars["VER"].recent_lap_times_ms == [91000.0, 92000.0]

    lap_14 = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=14,
            track_status=TrackStatus.GREEN,
            car_updates=[
                CarLapUpdate(
                    driver_id="VER",
                    team="Red Bull",
                    position=2,
                    lap_time_ms=104000.0,
                    gap_to_leader_ms=4500.0,
                    interval_ahead_ms=4500.0,
                    tyre_compound=TyreCompound.INTER,
                    pit_out=True,
                ),
                CarLapUpdate(
                    driver_id="LEC",
                    team="Ferrari",
                    position=1,
                    lap_time_ms=92100.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.MEDIUM,
                    tyre_age_laps=9,
                ),
            ],
        )
    )

    assert lap_14.cars["VER"].used_dry_compounds == {"MEDIUM", "HARD"}
    assert lap_14.cars["VER"].used_wet is True


def test_state_engine_marks_dirty_laps_and_feature_builder_uses_clean_history() -> None:
    engine = StateEngine(clean_lap_window=4, traffic_gap_threshold_ms=700.0)

    engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=1,
            track_status=TrackStatus.GREEN,
            weather={"air_c": 23.5, "track_c": 31.0, "humidity": 50.0},
            car_updates=[
                CarLapUpdate(
                    driver_id="NOR",
                    team="McLaren",
                    position=1,
                    lap_time_ms=90500.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.SOFT,
                )
            ],
        )
    )
    engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=2,
            track_status=TrackStatus.SC,
            car_updates=[
                CarLapUpdate(
                    driver_id="NOR",
                    team="McLaren",
                    position=1,
                    lap_time_ms=120000.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.SOFT,
                )
            ],
        )
    )
    engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=3,
            track_status=TrackStatus.GREEN,
            car_updates=[
                CarLapUpdate(
                    driver_id="NOR",
                    team="McLaren",
                    position=1,
                    lap_time_ms=93000.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    interval_behind_ms=450.0,
                    tyre_compound=TyreCompound.SOFT,
                ),
                CarLapUpdate(
                    driver_id="RUS",
                    team="Mercedes",
                    position=2,
                    lap_time_ms=93450.0,
                    gap_to_leader_ms=450.0,
                    interval_ahead_ms=450.0,
                    tyre_compound=TyreCompound.SOFT,
                ),
            ],
        )
    )
    state = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=4,
            track_status=TrackStatus.GREEN,
            weather={"air_c": 24.0},
            car_updates=[
                CarLapUpdate(
                    driver_id="NOR",
                    team="McLaren",
                    position=1,
                    lap_time_ms=90100.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    interval_behind_ms=1200.0,
                    tyre_compound=TyreCompound.SOFT,
                ),
                CarLapUpdate(
                    driver_id="RUS",
                    team="Mercedes",
                    position=2,
                    lap_time_ms=91300.0,
                    gap_to_leader_ms=1200.0,
                    interval_ahead_ms=1200.0,
                    tyre_compound=TyreCompound.SOFT,
                ),
            ],
        )
    )

    assert state.track_status is TrackStatus.GREEN
    assert state.cars["NOR"].recent_lap_times_ms == [90500.0, 90100.0]
    assert state.cars["NOR"].cleaning_flags.is_clean is True
    assert state.weather["air_c"] == 24.0
    assert state.weather["track_c"] == 31.0

    dirty_sc_state = engine.step(
        LapEndTick(
            session_id="2023_test_r",
            lap=5,
            car_updates=[
                CarLapUpdate(
                    driver_id="NOR",
                    team="McLaren",
                    position=1,
                    lap_time_ms=89900.0,
                    gap_to_leader_ms=0.0,
                    interval_ahead_ms=0.0,
                    tyre_compound=TyreCompound.SOFT,
                    track_status="6",
                )
            ],
        )
    )
    assert dirty_sc_state.track_status is TrackStatus.VSC
    assert dirty_sc_state.cars["NOR"].cleaning_flags.is_sc_vsc is True
    assert dirty_sc_state.cars["NOR"].recent_lap_times_ms == [90500.0, 90100.0]

    features = build_driver_features(dirty_sc_state, "NOR")
    assert features["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert features["recent_clean_lap_times_ms"] == [90500.0, 90100.0]
    assert features["cleaning_flags"]["is_sc_vsc"] is True
