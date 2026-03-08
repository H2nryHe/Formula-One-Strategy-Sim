"""Explainable baseline models for replay-only MVP evaluation."""

from __future__ import annotations

import random
from statistics import median
from typing import Any

from f1sim.contracts import DegradationPred, LapTimePred, PitProbPred, Scenario
from f1sim.features import FEATURE_SCHEMA_VERSION
from f1sim.models.config import (
    DegradationModelV0Config,
    PaceModelV0Config,
    PitPolicyModelV0Config,
    ScenarioModelV0Config,
)
from f1sim.state import RaceState, TrackStatus, TyreCompound


class DegradationModelV0:
    model_name = "degradation_v0"
    model_version = "degradation_v0.1"
    trained_on = "heuristic"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(self, config: DegradationModelV0Config | None = None) -> None:
        self.config = config or DegradationModelV0Config()

    def predict_delta(
        self,
        compound: str,
        tyre_age_laps: int,
        ctx: dict[str, object],
    ) -> DegradationPred:
        profile = self.config.compounds.get(compound.upper(), self.config.compounds["UNKNOWN"])
        knee_age = profile["knee_age"]
        early_age = min(tyre_age_laps, knee_age)
        late_age = max(0, tyre_age_laps - knee_age)
        delta_mean = early_age * profile["slope_1"] + late_age * profile["slope_2"]
        delta_sigma = self.config.base_sigma_ms + tyre_age_laps * 6.0
        cliff_risk = _sigmoid((tyre_age_laps - profile["cliff_age"]) / 2.0)
        return DegradationPred(
            delta_mean_ms=delta_mean,
            delta_sigma_ms=delta_sigma,
            cliff_risk=cliff_risk,
        )


class PaceModelV0:
    model_name = "pace_v0"
    model_version = "pace_v0.1"
    trained_on = "heuristic"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(
        self,
        *,
        degradation_model: DegradationModelV0 | None = None,
        config: PaceModelV0Config | None = None,
    ) -> None:
        self.degradation_model = degradation_model or DegradationModelV0()
        self.config = config or PaceModelV0Config()

    def predict_lap_time(
        self,
        state: RaceState,
        driver_id: str,
        *,
        horizon_lap: int = 1,
    ) -> LapTimePred:
        car = state.cars[driver_id]
        recent_clean = car.recent_lap_times_ms
        if recent_clean:
            base_mean = median(recent_clean)
        elif car.last_lap_time_ms is not None:
            base_mean = car.last_lap_time_ms
        else:
            base_mean = _field_median(state)

        stint_age = max(0, car.tyre_age_laps + horizon_lap - 1)
        degradation = self.degradation_model.predict_delta(
            car.tyre_compound.value,
            stint_age,
            {"track_status": state.track_status.value, "weather": state.weather},
        )
        track_key = state.track_status.value
        track_adj = self.config.track_status_adjustments_ms.get(track_key, 0.0)
        traffic_adj = _traffic_adjustment(car=car, config=self.config)
        weather_adj = _weather_adjustment(weather=state.weather, config=self.config)
        spread_sigma = _sample_sigma(recent_clean)
        sigma = max(
            self.config.min_sigma_ms,
            spread_sigma
            + degradation.delta_sigma_ms
            + self.config.track_status_sigma_ms.get(track_key, 0.0),
        )
        components = {
            "base": base_mean,
            "degradation": degradation.delta_mean_ms,
            "track_status": track_adj,
            "traffic": traffic_adj,
            "weather": weather_adj,
        }
        return LapTimePred(
            mean_ms=base_mean + degradation.delta_mean_ms + track_adj + traffic_adj + weather_adj,
            sigma_ms=sigma,
            components=components,
        )


class PitPolicyModelV0:
    model_name = "pit_policy_v0"
    model_version = "pit_policy_v0.1"
    trained_on = "heuristic"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(self, config: PitPolicyModelV0Config | None = None) -> None:
        self.config = config or PitPolicyModelV0Config()

    def predict_pit_prob(
        self,
        state: RaceState,
        driver_id: str,
        *,
        window_laps: int = 1,
    ) -> PitProbPred:
        car = state.cars[driver_id]
        baseline = (
            median(car.recent_lap_times_ms)
            if car.recent_lap_times_ms
            else (car.last_lap_time_ms or 90000.0)
        )
        pace_drop = (
            0.0
            if car.last_lap_time_ms is None
            else max(0.0, car.last_lap_time_ms - baseline)
        )
        threshold = self.config.compound_thresholds.get(
            car.tyre_compound.value,
            self.config.compound_thresholds["UNKNOWN"],
        )
        age_pressure = car.tyre_age_laps - threshold
        gaps = [
            gap
            for gap in (car.interval_ahead_ms, car.interval_behind_ms)
            if gap is not None and gap > 0
        ]
        min_gap = min(gaps) if gaps else None
        score = self.config.base_bias
        score += self.config.age_weight * age_pressure
        score += self.config.pace_drop_weight * pace_drop
        score += self.config.track_status_bias.get(state.track_status.value, 0.0)
        if car.cleaning_flags.is_traffic_heavy:
            score += self.config.traffic_weight
        if min_gap is not None:
            score += self.config.gap_weight * max(0.0, 1500.0 - min_gap)

        per_lap_prob = _sigmoid(score)
        p_window = 1.0 - (1.0 - per_lap_prob) ** max(window_laps, 1)
        return PitProbPred(
            p_pit_in_window=p_window,
            p_compound=_pit_compound_distribution(car.tyre_compound),
            calibration_meta={
                "score": f"{score:.4f}",
                "per_lap_prob": f"{per_lap_prob:.4f}",
                "threshold": str(threshold),
            },
        )


class ScenarioModelV0:
    model_name = "scenario_v0"
    model_version = "scenario_v0.1"
    trained_on = "heuristic"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(self, config: ScenarioModelV0Config | None = None) -> None:
        self.config = config or ScenarioModelV0Config()

    def sample_scenarios(
        self,
        state: RaceState,
        *,
        horizon_laps: int,
        n: int,
        seed: int,
    ) -> list[Scenario]:
        rng = random.Random(seed)
        scenarios: list[Scenario] = []
        for _index in range(n):
            status_path = _build_track_status_path(
                state=state,
                horizon_laps=horizon_laps,
                rng=rng,
                config=self.config,
            )
            weather_path = [dict(state.weather) for _ in range(horizon_laps)]
            scenarios.append(Scenario(track_status_path=status_path, weather_path=weather_path))
        return scenarios


def _field_median(state: RaceState) -> float:
    values = [
        car.last_lap_time_ms
        for car in state.cars.values()
        if car.last_lap_time_ms is not None
    ]
    return median(values) if values else 90000.0


def _sample_sigma(values: list[float]) -> float:
    if len(values) < 2:
        return 140.0
    center = median(values)
    deviations = [abs(value - center) for value in values]
    return max(120.0, median(deviations) * 1.4826)


def _traffic_adjustment(*, car: Any, config: PaceModelV0Config) -> float:
    gaps = [
        gap
        for gap in (car.interval_ahead_ms, car.interval_behind_ms)
        if gap is not None and gap > 0
    ]
    if not gaps:
        return 0.0
    min_gap = min(gaps)
    if min_gap >= config.traffic_gap_threshold_ms:
        return 0.0
    scale = 1.0 - (min_gap / config.traffic_gap_threshold_ms)
    return config.max_traffic_penalty_ms * scale


def _weather_adjustment(*, weather: dict[str, float | None], config: PaceModelV0Config) -> float:
    rainfall = weather.get("rainfall")
    track_temp = weather.get("track_c")
    adjustment = 0.0
    if rainfall is not None:
        adjustment += rainfall * config.rainfall_penalty_ms_per_unit
    if track_temp is not None:
        delta = track_temp - config.ideal_track_temp_c
        if delta < 0:
            adjustment += abs(delta) * config.cold_track_penalty_ms_per_c
        else:
            adjustment += delta * config.hot_track_penalty_ms_per_c
    return adjustment


def _pit_compound_distribution(current_compound: TyreCompound) -> dict[str, float]:
    if current_compound is TyreCompound.SOFT:
        return {"MEDIUM": 0.7, "HARD": 0.3}
    if current_compound is TyreCompound.MEDIUM:
        return {"HARD": 0.65, "SOFT": 0.35}
    if current_compound is TyreCompound.HARD:
        return {"MEDIUM": 0.6, "SOFT": 0.4}
    return {"HARD": 1.0}


def _build_track_status_path(
    *,
    state: RaceState,
    horizon_laps: int,
    rng: random.Random,
    config: ScenarioModelV0Config,
) -> list[str]:
    current = state.track_status.value
    if state.track_status in {TrackStatus.SC, TrackStatus.VSC}:
        persistence = 1 + rng.randrange(max(1, min(config.persistence_cap_laps, horizon_laps)))
        path = [
            current if lap_index < persistence else config.green_status_name
            for lap_index in range(horizon_laps)
        ]
        return path
    if state.track_status is TrackStatus.GREEN and horizon_laps > 0:
        if rng.random() < config.vsc_transition_probability:
            trigger_lap = rng.randrange(horizon_laps)
            path = [config.green_status_name for _ in range(horizon_laps)]
            path[trigger_lap] = TrackStatus.VSC.value
            return path
    return [current for _ in range(horizon_laps)]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = pow(2.718281828459045, -value)
        return 1.0 / (1.0 + exp_term)
    exp_term = pow(2.718281828459045, value)
    return exp_term / (1.0 + exp_term)
