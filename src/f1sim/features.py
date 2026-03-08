"""Versioned feature-builder stubs for lap-end replay ticks."""

from __future__ import annotations

from f1sim.state import RaceState

FEATURE_SCHEMA_VERSION = "0.2.0"


def build_driver_features(state: RaceState, driver_id: str) -> dict[str, object]:
    car = state.cars[driver_id]
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "session_id": state.session_id,
        "lap": state.lap,
        "driver_id": driver_id,
        "track_status": state.track_status.value,
        "position": car.position,
        "gap_to_leader_ms": car.gap_to_leader_ms,
        "interval_ahead_ms": car.interval_ahead_ms,
        "interval_behind_ms": car.interval_behind_ms,
        "tyre_compound": car.tyre_compound.value,
        "tyre_age_laps": car.tyre_age_laps,
        "stint_id": car.stint_id,
        "used_dry_compounds": sorted(car.used_dry_compounds),
        "used_wet": car.used_wet,
        "is_pitting": car.is_pitting,
        "pit_in": car.pit_in,
        "pit_out": car.pit_out,
        "last_lap_time_ms": car.last_lap_time_ms,
        "recent_clean_lap_times_ms": list(car.recent_lap_times_ms),
        "weather": dict(state.weather),
        "cleaning_flags": car.cleaning_flags.to_dict(),
    }


def build_feature_frame(state: RaceState) -> dict[str, dict[str, object]]:
    return {driver_id: build_driver_features(state, driver_id) for driver_id in sorted(state.cars)}
