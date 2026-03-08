from f1sim import TrackStatus, TyreCompound, default_assumptions_hash
from f1sim.replay import build_bootstrap_state


def test_stage0_bootstrap_smoke() -> None:
    state = build_bootstrap_state()

    assert state.session_id == "bootstrap-demo-2023-monza-r"
    assert state.lap == 0
    assert state.track_status is TrackStatus.GREEN
    assert set(state.cars) == {"LEC", "VER"}
    assert state.cars["VER"].tyre_compound is TyreCompound.MEDIUM
    assert len(default_assumptions_hash()) == 64
