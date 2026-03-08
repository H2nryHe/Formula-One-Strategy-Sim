"""Configuration for baseline v0 replay models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DegradationModelV0Config:
    base_sigma_ms: float = 80.0
    compounds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "SOFT": {"slope_1": 85.0, "slope_2": 165.0, "cliff_age": 14.0, "knee_age": 10.0},
            "MEDIUM": {"slope_1": 55.0, "slope_2": 120.0, "cliff_age": 20.0, "knee_age": 16.0},
            "HARD": {"slope_1": 35.0, "slope_2": 85.0, "cliff_age": 28.0, "knee_age": 22.0},
            "INTER": {"slope_1": 90.0, "slope_2": 170.0, "cliff_age": 10.0, "knee_age": 7.0},
            "WET": {"slope_1": 95.0, "slope_2": 175.0, "cliff_age": 10.0, "knee_age": 7.0},
            "UNKNOWN": {"slope_1": 50.0, "slope_2": 110.0, "cliff_age": 18.0, "knee_age": 14.0},
        }
    )


@dataclass(slots=True)
class PaceModelV0Config:
    min_sigma_ms: float = 180.0
    track_status_adjustments_ms: dict[str, float] = field(
        default_factory=lambda: {
            "GREEN": 0.0,
            "YELLOW": 4000.0,
            "VSC": 18000.0,
            "SC": 30000.0,
            "RED": 60000.0,
        }
    )
    track_status_sigma_ms: dict[str, float] = field(
        default_factory=lambda: {
            "GREEN": 0.0,
            "YELLOW": 150.0,
            "VSC": 350.0,
            "SC": 500.0,
            "RED": 700.0,
        }
    )
    traffic_gap_threshold_ms: float = 1200.0
    max_traffic_penalty_ms: float = 900.0
    rainfall_penalty_ms_per_unit: float = 2500.0
    cold_track_penalty_ms_per_c: float = 12.0
    hot_track_penalty_ms_per_c: float = 7.0
    ideal_track_temp_c: float = 33.0


@dataclass(slots=True)
class PitPolicyModelV0Config:
    base_bias: float = -2.0
    age_weight: float = 0.32
    pace_drop_weight: float = 0.0013
    traffic_weight: float = 0.35
    gap_weight: float = 0.00015
    track_status_bias: dict[str, float] = field(
        default_factory=lambda: {
            "GREEN": 0.0,
            "YELLOW": 0.15,
            "VSC": 1.1,
            "SC": 1.45,
            "RED": -4.0,
        }
    )
    compound_thresholds: dict[str, int] = field(
        default_factory=lambda: {
            "SOFT": 14,
            "MEDIUM": 20,
            "HARD": 28,
            "INTER": 10,
            "WET": 10,
            "UNKNOWN": 18,
        }
    )


@dataclass(slots=True)
class ScenarioModelV0Config:
    green_status_name: str = "GREEN"
    persistence_cap_laps: int = 4
    vsc_transition_probability: float = 0.15
