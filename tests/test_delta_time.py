from f1sim.metrics.delta_time import (
    accumulate_plan_total_time,
    compute_delta_time,
    suspicion_reason_for_delta_time,
)


def test_compute_delta_time_positive_means_plan_faster() -> None:
    assert compute_delta_time(1000.0, 1100.0) == 100.0


def test_accumulate_plan_total_time_includes_pit_loss() -> None:
    total = accumulate_plan_total_time(
        [1000.0, 1000.0, 1000.0],
        pit_loss_ms=20000.0,
    )
    assert total == 23000.0


def test_suspicion_reason_flags_absurd_delta() -> None:
    reason = suspicion_reason_for_delta_time(
        delta_time_ms=-200000.0,
        horizon_laps=10,
    )
    assert reason is not None
    assert "exceeds cap" in reason
