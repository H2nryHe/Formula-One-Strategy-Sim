from f1sim.models import DegradationModelV0, PaceModelV0, PitPolicyModelV0, ScenarioModelV0
from f1sim.state import CarState, CleaningFlags, RaceState, TrackStatus, TyreCompound


def _build_state(
    *,
    track_status: TrackStatus = TrackStatus.GREEN,
    interval_behind_ms: float | None = 1500.0,
    tyre_age_laps: int = 18,
) -> RaceState:
    return RaceState(
        session_id="model-test",
        lap=25,
        track_status=track_status,
        weather={"track_c": 31.0, "rainfall": 0.0},
        cars={
            "VER": CarState(
                driver_id="VER",
                team="Red Bull",
                position=1,
                gap_to_leader_ms=0.0,
                interval_ahead_ms=0.0,
                interval_behind_ms=interval_behind_ms,
                tyre_compound=TyreCompound.MEDIUM,
                tyre_age_laps=tyre_age_laps,
                stint_id=1,
                recent_lap_times_ms=[90200.0, 90500.0, 90400.0],
                last_lap_time_ms=90750.0,
                cleaning_flags=CleaningFlags(
                    is_sc_vsc=track_status in {TrackStatus.SC, TrackStatus.VSC},
                    is_traffic_heavy=interval_behind_ms is not None and interval_behind_ms < 1000.0,
                ),
            )
        },
    )


def test_pace_model_v0_is_state_sensitive() -> None:
    degradation_model = DegradationModelV0()
    pace_model = PaceModelV0(degradation_model=degradation_model)

    clean_pred = pace_model.predict_lap_time(_build_state(), "VER")
    traffic_pred = pace_model.predict_lap_time(_build_state(interval_behind_ms=400.0), "VER")
    sc_pred = pace_model.predict_lap_time(_build_state(track_status=TrackStatus.SC), "VER")

    assert clean_pred.mean_ms > 0
    assert clean_pred.sigma_ms > 0
    assert traffic_pred.mean_ms > clean_pred.mean_ms
    assert sc_pred.mean_ms > traffic_pred.mean_ms
    assert clean_pred.components["degradation"] > 0


def test_pit_policy_model_v0_probabilities_are_non_degenerate() -> None:
    pit_model = PitPolicyModelV0()

    fresh_state = _build_state(tyre_age_laps=6)
    worn_state = _build_state(tyre_age_laps=23, interval_behind_ms=500.0)

    p1_fresh = pit_model.predict_pit_prob(fresh_state, "VER", window_laps=1)
    p3_fresh = pit_model.predict_pit_prob(fresh_state, "VER", window_laps=3)
    p3_worn = pit_model.predict_pit_prob(worn_state, "VER", window_laps=3)

    assert 0.0 <= p1_fresh.p_pit_in_window <= 1.0
    assert p3_fresh.p_pit_in_window >= p1_fresh.p_pit_in_window
    assert p3_worn.p_pit_in_window > p3_fresh.p_pit_in_window
    assert p3_worn.calibration_meta["threshold"] == "20"


def test_degradation_and_scenarios_are_seeded_and_non_constant() -> None:
    degradation_model = DegradationModelV0()
    scenario_model = ScenarioModelV0()

    newer = degradation_model.predict_delta("MEDIUM", 5, {})
    older = degradation_model.predict_delta("MEDIUM", 22, {})
    state = _build_state(track_status=TrackStatus.SC)

    scenarios_a = scenario_model.sample_scenarios(state, horizon_laps=4, n=3, seed=7)
    scenarios_b = scenario_model.sample_scenarios(state, horizon_laps=4, n=3, seed=7)

    assert older.delta_mean_ms > newer.delta_mean_ms
    assert older.cliff_risk > newer.cliff_risk
    assert scenarios_a == scenarios_b
    assert any(
        path.track_status_path != scenarios_a[0].track_status_path
        for path in scenarios_a[1:]
    )
