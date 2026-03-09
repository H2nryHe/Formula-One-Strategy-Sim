"""Single-source definitions for delta-time semantics and sanity checks."""

from __future__ import annotations

import math

DELTA_TIME_FORMULA = "baseline_total_time_ms - plan_total_time_ms"
DELTA_TIME_DEFINITION_LABEL = (
    "Positive Δtime means the plan is faster than the baseline; negative means slower."
)


def compute_delta_time(plan_time_ms: float, baseline_time_ms: float) -> float:
    """Return plan gain vs baseline in milliseconds.

    Definition:
      Δtime_ms = baseline_total_time_ms - plan_total_time_ms

    Positive values mean the plan is faster than the baseline.
    """

    _assert_time_ms(plan_time_ms, name="plan_time_ms")
    _assert_time_ms(baseline_time_ms, name="baseline_time_ms")
    return baseline_time_ms - plan_time_ms


def accumulate_plan_total_time(
    per_lap_times_ms: list[float],
    *,
    pit_loss_ms: float = 0.0,
) -> float:
    _assert_time_ms(pit_loss_ms, name="pit_loss_ms", allow_zero=True)
    total = pit_loss_ms
    for lap_time_ms in per_lap_times_ms:
        _assert_time_ms(lap_time_ms, name="lap_time_ms")
        total += lap_time_ms
    return total


def delta_time_cap_ms(
    *,
    horizon_laps: int,
    delta_time_per_lap_cap_ms: float = 6000.0,
) -> float:
    if horizon_laps <= 0:
        raise ValueError("horizon_laps must be positive")
    _assert_time_ms(
        delta_time_per_lap_cap_ms,
        name="delta_time_per_lap_cap_ms",
        allow_zero=False,
    )
    return horizon_laps * delta_time_per_lap_cap_ms


def suspicion_reason_for_delta_time(
    *,
    delta_time_ms: float,
    horizon_laps: int,
    delta_time_per_lap_cap_ms: float = 6000.0,
) -> str | None:
    cap_ms = delta_time_cap_ms(
        horizon_laps=horizon_laps,
        delta_time_per_lap_cap_ms=delta_time_per_lap_cap_ms,
    )
    if abs(delta_time_ms) <= cap_ms:
        return None
    return (
        f"|Δtime|={abs(delta_time_ms):.1f} ms exceeds cap {cap_ms:.1f} ms "
        f"for horizon={horizon_laps}"
    )


def _assert_time_ms(value: float, *, name: str, allow_zero: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative milliseconds")
    if value == 0.0 and not allow_zero:
        raise ValueError(f"{name} must be positive milliseconds")
