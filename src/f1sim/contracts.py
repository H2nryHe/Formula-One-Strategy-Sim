"""Typed outputs and interfaces for pluggable strategy models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from f1sim.state import RaceState


class ActionType(str, Enum):
    STAY_OUT = "STAY_OUT"
    PIT_TO_SOFT = "PIT_TO_SOFT"
    PIT_TO_MEDIUM = "PIT_TO_MEDIUM"
    PIT_TO_HARD = "PIT_TO_HARD"
    PIT_TO_INTER = "PIT_TO_INTER"
    PIT_TO_WET = "PIT_TO_WET"


@dataclass(slots=True)
class LapTimePred:
    mean_ms: float
    sigma_ms: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DegradationPred:
    delta_mean_ms: float
    delta_sigma_ms: float
    cliff_risk: float | None = None


@dataclass(slots=True)
class PitProbPred:
    p_pit_in_window: float
    p_compound: dict[str, float] = field(default_factory=dict)
    calibration_meta: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Scenario:
    track_status_path: list[str] = field(default_factory=list)
    weather_path: list[dict[str, float | None]] = field(default_factory=list)


@dataclass(slots=True)
class Explanation:
    code: str
    text: str
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PlanMetrics:
    delta_time_mean_ms: float
    delta_time_p10_ms: float
    delta_time_p50_ms: float
    delta_time_p90_ms: float
    p_gain_pos_ge_1: float
    risk_sigma_ms: float


@dataclass(slots=True)
class Plan:
    plan_id: str
    actions: list[dict[str, int | str]]
    metrics: PlanMetrics
    explanations: list[Explanation] = field(default_factory=list)
    counterfactuals: dict[str, dict[str, float | str]] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    is_suspicious: bool = False
    suspicion_reason: str | None = None


@dataclass(slots=True)
class PlanComparison:
    baseline_plan_id: str
    delta_time_mean_ms: float
    notes: str = ""


@dataclass(slots=True)
class RecommendationBundle:
    session_id: str
    lap: int
    target_driver: str
    generated_at_ts: str
    top_k: list[Plan]
    baselines: dict[str, PlanComparison] = field(default_factory=dict)
    ground_truth: dict[str, object] = field(default_factory=dict)
    assumptions_hash: str = ""
    model_versions: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class PaceModel(Protocol):
    model_name: str
    model_version: str
    trained_on: str
    feature_schema_version: str

    def predict_lap_time(
        self,
        state: RaceState,
        driver_id: str,
        *,
        horizon_lap: int = 1,
    ) -> LapTimePred: ...


class DegradationModel(Protocol):
    model_name: str
    model_version: str
    trained_on: str
    feature_schema_version: str

    def predict_delta(
        self,
        compound: str,
        tyre_age_laps: int,
        ctx: dict[str, object],
    ) -> DegradationPred: ...


class PitPolicyModel(Protocol):
    model_name: str
    model_version: str
    trained_on: str
    feature_schema_version: str

    def predict_pit_prob(
        self,
        state: RaceState,
        driver_id: str,
        *,
        window_laps: int = 1,
    ) -> PitProbPred: ...


class ScenarioModel(Protocol):
    model_name: str
    model_version: str
    trained_on: str
    feature_schema_version: str

    def sample_scenarios(
        self,
        state: RaceState,
        *,
        horizon_laps: int,
        n: int,
        seed: int,
    ) -> list[Scenario]: ...


class StrategySearcher(Protocol):
    model_name: str
    model_version: str
    trained_on: str
    feature_schema_version: str

    def recommend(
        self,
        state: RaceState,
        target_driver: str,
        *,
        horizon_laps: int,
        top_k: int,
        seed: int,
    ) -> RecommendationBundle: ...
