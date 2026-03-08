"""Helpers for Stage 0 replay smoke checks."""

from __future__ import annotations

from f1sim.state import CarState, CleaningFlags, RaceState, TrackStatus, TyreCompound


def build_bootstrap_state() -> RaceState:
    """Return a small deterministic replay state for import/smoke coverage."""
    return RaceState(
        session_id="bootstrap-demo-2023-monza-r",
        lap=0,
        track_status=TrackStatus.GREEN,
        total_laps=53,
        weather={
            "air_c": 24.0,
            "track_c": 33.0,
            "humidity": 52.0,
            "rainfall": 0.0,
            "wind_ms": 1.8,
        },
        cars={
            "VER": CarState(
                driver_id="VER",
                team="Red Bull",
                position=1,
                gap_to_leader_ms=0.0,
                interval_ahead_ms=0.0,
                interval_behind_ms=1750.0,
                tyre_compound=TyreCompound.MEDIUM,
                tyre_age_laps=0,
                stint_id=0,
                used_dry_compounds={"MEDIUM"},
                recent_lap_times_ms=[],
                last_lap_time_ms=None,
                cleaning_flags=CleaningFlags(has_valid_lap_time=False),
            ),
            "LEC": CarState(
                driver_id="LEC",
                team="Ferrari",
                position=2,
                gap_to_leader_ms=1750.0,
                interval_ahead_ms=1750.0,
                interval_behind_ms=640.0,
                tyre_compound=TyreCompound.MEDIUM,
                tyre_age_laps=0,
                stint_id=0,
                used_dry_compounds={"MEDIUM"},
                recent_lap_times_ms=[],
                last_lap_time_ms=None,
                cleaning_flags=CleaningFlags(has_valid_lap_time=False),
            ),
        },
    )
