"""Rollout-based strategy search for replay-first recommendations."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace
from statistics import median
from typing import Any

from f1sim.assumptions import default_assumptions_hash
from f1sim.contracts import (
    Plan,
    PlanComparison,
    PlanMetrics,
    RecommendationBundle,
    Scenario,
)
from f1sim.eval.metrics import quantile
from f1sim.explainer import build_plan_counterfactuals, build_plan_explanations
from f1sim.features import FEATURE_SCHEMA_VERSION
from f1sim.metrics import (
    accumulate_plan_total_time,
    compute_delta_time,
    contribution_breakdown,
    delta_time_cap_ms,
    suspicion_reason_for_delta_time,
)
from f1sim.models import (
    DegradationModelV0,
    PaceModelV0,
    PitPolicyModelV0,
    PitPolicyModelV0Config,
    ScenarioModelV0,
)
from f1sim.rules import action_mask, plan_satisfies_rules
from f1sim.state import CarState, RaceState, TrackStatus, TyreCompound

DEFAULT_RULE_THRESHOLDS = {
    TyreCompound.SOFT.value: 14,
    TyreCompound.MEDIUM.value: 20,
    TyreCompound.HARD.value: 28,
    TyreCompound.INTER.value: 10,
    TyreCompound.WET.value: 10,
    TyreCompound.UNKNOWN.value: 18,
}

SEARCH_MODEL_VERSION = "rollout_search_v0.3"


@dataclass(slots=True)
class ModelSuite:
    degradation_model: DegradationModelV0
    pace_model: PaceModelV0
    pit_policy_model: PitPolicyModelV0
    scenario_model: ScenarioModelV0

    def model_versions(self) -> dict[str, str]:
        return {
            "pace": self.pace_model.model_version,
            "degradation": self.degradation_model.model_version,
            "pit_policy": self.pit_policy_model.model_version,
            "scenario": self.scenario_model.model_version,
            "search": SEARCH_MODEL_VERSION,
        }


@dataclass(slots=True)
class RolloutSearchConfig:
    horizon_laps: int = 8
    n_scenarios: int = 16
    copy_policy: str = "nearest"
    top_k: int = 3
    two_dry_deadline_laps: int = 12
    sanity_delta_time_per_lap_cap_ms: float = 6000.0
    rule_thresholds: dict[str, int] | None = None

    def resolved_thresholds(self) -> dict[str, int]:
        thresholds = dict(DEFAULT_RULE_THRESHOLDS)
        if self.rule_thresholds is not None:
            thresholds.update(self.rule_thresholds)
        return thresholds


@dataclass(slots=True)
class SimulationOutcome:
    total_time_ms: float
    final_position: int
    pit_loss_ms_used: float
    per_lap_times_ms: list[float]
    pace_component_totals_ms: dict[str, float]


@dataclass(frozen=True, slots=True)
class CandidatePlanSpec:
    plan_id: str
    actions: tuple[dict[str, int | str], ...]


def build_model_suite(rule_thresholds: dict[str, int] | None = None) -> ModelSuite:
    thresholds = dict(DEFAULT_RULE_THRESHOLDS)
    if rule_thresholds is not None:
        thresholds.update(rule_thresholds)
    degradation_model = DegradationModelV0()
    pace_model = PaceModelV0(degradation_model=degradation_model)
    pit_policy_model = PitPolicyModelV0(
        config=PitPolicyModelV0Config(compound_thresholds=dict(thresholds))
    )
    scenario_model = ScenarioModelV0()
    return ModelSuite(
        degradation_model=degradation_model,
        pace_model=pace_model,
        pit_policy_model=pit_policy_model,
        scenario_model=scenario_model,
    )


class RolloutStrategySearcher:
    """Evaluate a small action set with deterministic rollout simulation."""

    model_name = "rollout_search_v0"
    model_version = SEARCH_MODEL_VERSION
    trained_on = "heuristic"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def __init__(
        self,
        *,
        suite: ModelSuite | None = None,
        config: RolloutSearchConfig | None = None,
    ) -> None:
        self.config = config or RolloutSearchConfig()
        self.suite = suite or build_model_suite(self.config.rule_thresholds)

    def recommend(
        self,
        state: RaceState,
        target_driver: str,
        *,
        horizon_laps: int,
        top_k: int,
        seed: int,
    ) -> RecommendationBundle:
        bundle, _ = self._recommend_internal(
            state=state,
            target_driver=target_driver,
            horizon_laps=horizon_laps,
            top_k=top_k,
            seed=seed,
        )
        return bundle

    def recommend_with_artifacts(
        self,
        state: RaceState,
        target_driver: str,
        *,
        horizon_laps: int | None = None,
        top_k: int | None = None,
        seed: int = 7,
    ) -> tuple[RecommendationBundle, dict[str, Plan]]:
        return self._recommend_internal(
            state=state,
            target_driver=target_driver,
            horizon_laps=horizon_laps or self.config.horizon_laps,
            top_k=top_k or self.config.top_k,
            seed=seed,
        )

    def _recommend_internal(
        self,
        *,
        state: RaceState,
        target_driver: str,
        horizon_laps: int,
        top_k: int,
        seed: int,
    ) -> tuple[RecommendationBundle, dict[str, Plan]]:
        if target_driver not in state.cars:
            raise ValueError(f"target driver not present in state: {target_driver}")

        thresholds = self.config.resolved_thresholds()
        race_total_laps = _race_total_laps(state=state, fallback_horizon=horizon_laps)
        scenarios = self.suite.scenario_model.sample_scenarios(
            state,
            horizon_laps=horizon_laps,
            n=self.config.n_scenarios,
            seed=seed,
        )
        candidate_plans = self._candidate_plans(
            state=state,
            target_driver=target_driver,
            horizon_laps=horizon_laps,
            race_total_laps=race_total_laps,
        )
        stay_out_outcomes = self._simulate_plan(
            state=state,
            target_driver=target_driver,
            plan_actions=(),
            scenarios=scenarios,
            seed=seed,
        )

        plan_outcomes: dict[str, list[SimulationOutcome]] = {"STAY_OUT": stay_out_outcomes}
        for plan_spec in candidate_plans:
            if plan_spec.plan_id == "STAY_OUT":
                continue
            plan_outcomes[plan_spec.plan_id] = self._simulate_plan(
                state=state,
                target_driver=target_driver,
                plan_actions=plan_spec.actions,
                scenarios=scenarios,
                seed=seed,
            )

        plans = {
            plan_spec.plan_id: self._plan_from_outcomes(
                state=state,
                target_driver=target_driver,
                plan_id=plan_spec.plan_id,
                actions=list(plan_spec.actions),
                outcomes=outcomes,
                baseline_outcomes=stay_out_outcomes,
                seed=seed,
            )
            for plan_spec in candidate_plans
            for outcomes in [plan_outcomes[plan_spec.plan_id]]
        }
        for plan in plans.values():
            plan.explanations = build_plan_explanations(
                state=state,
                target_driver=target_driver,
                plan=plan,
                plans_by_id=plans,
                degradation_model=self.suite.degradation_model,
                pit_policy_model=self.suite.pit_policy_model,
                n_scenarios=self.config.n_scenarios,
                race_total_laps=race_total_laps,
                deadline_laps=self.config.two_dry_deadline_laps,
            )
            plan.counterfactuals = build_plan_counterfactuals(plan=plan, plans_by_id=plans)
        feasible_plan_ids = [
            plan_spec.plan_id
            for plan_spec in candidate_plans
            if self._plan_is_feasible(
                state=state,
                target_driver=target_driver,
                plan=plans[plan_spec.plan_id],
                race_total_laps=race_total_laps,
            )
        ]
        ranked = sorted(
            (plans[action] for action in feasible_plan_ids),
            key=lambda plan: (plan.metrics.delta_time_mean_ms, -plan.metrics.risk_sigma_ms),
            reverse=True,
        )

        if top_k > len(ranked):
            warnings = [f"requested top_k={top_k} but only {len(ranked)} feasible plans available"]
            selected = ranked
        else:
            warnings = []
            selected = ranked[:top_k]

        baseline_plans = {
            "STAY_OUT": plans["STAY_OUT"],
            "RULE_TYRE_AGE": plans[
                self._resolve_baseline_plan_id(
                    plans=plans,
                    action=self._rule_tyre_age_action(state, target_driver, thresholds),
                )
            ],
            f"COPY_{self.config.copy_policy.upper()}": plans[
                self._resolve_baseline_plan_id(
                    plans=plans,
                    action=self._copy_action(state, target_driver, self.config.copy_policy),
                )
            ],
        }
        best_plan = selected[0]
        baselines = {
            name: _plan_comparison(best_plan, baseline_plan)
            for name, baseline_plan in baseline_plans.items()
        }
        bundle = RecommendationBundle(
            session_id=state.session_id,
            lap=state.lap,
            target_driver=target_driver,
            generated_at_ts=f"replay:{state.session_id}:lap:{state.lap}:seed:{seed}",
            top_k=selected,
            baselines=baselines,
            assumptions_hash=default_assumptions_hash(),
            model_versions=self.suite.model_versions(),
            warnings=warnings,
        )
        return bundle, baseline_plans

    def _candidate_plans(
        self,
        *,
        state: RaceState,
        target_driver: str,
        horizon_laps: int,
        race_total_laps: int,
    ) -> list[CandidatePlanSpec]:
        actions = [
            "PIT_TO_SOFT",
            "PIT_TO_MEDIUM",
            "PIT_TO_HARD",
        ]
        rainfall = state.weather.get("rainfall")
        wet_context = (
            rainfall is not None and rainfall > 0.0
        ) or any(
            car.tyre_compound in {TyreCompound.INTER, TyreCompound.WET}
            for car in state.cars.values()
        )
        if wet_context:
            actions.extend(["PIT_TO_INTER", "PIT_TO_WET"])
        plans = [CandidatePlanSpec(plan_id="STAY_OUT", actions=())]
        max_delay = min(horizon_laps, max(0, race_total_laps - state.lap))
        for delay in range(1, max_delay + 1):
            planned_lap = state.lap + delay
            for action in actions:
                plan = Plan(
                    plan_id=_plan_id_for_scheduled_action(action, delay),
                    actions=_plan_actions(state.lap, action, delay_laps=delay),
                    metrics=PlanMetrics(
                        delta_time_mean_ms=0.0,
                        delta_time_p10_ms=0.0,
                        delta_time_p50_ms=0.0,
                        delta_time_p90_ms=0.0,
                        p_gain_pos_ge_1=0.0,
                        risk_sigma_ms=0.0,
                    ),
                )
                if not _plan_respects_rule_deadline(
                    state=state,
                    driver_id=target_driver,
                    plan=plan,
                    race_total_laps=race_total_laps,
                    deadline_laps=self.config.two_dry_deadline_laps,
                    planned_lap=planned_lap,
                ):
                    continue
                plans.append(
                    CandidatePlanSpec(
                        plan_id=plan.plan_id,
                        actions=tuple(plan.actions),
                    )
                )
        return plans

    def _plan_is_feasible(
        self,
        *,
        state: RaceState,
        target_driver: str,
        plan: Plan,
        race_total_laps: int,
    ) -> bool:
        return plan_satisfies_rules(state, target_driver, plan, race_total_laps) and (
            _plan_respects_rule_deadline(
                state=state,
                driver_id=target_driver,
                plan=plan,
                race_total_laps=race_total_laps,
                deadline_laps=self.config.two_dry_deadline_laps,
            )
        )

    @staticmethod
    def _resolve_baseline_plan_id(*, plans: dict[str, Plan], action: str) -> str:
        if action == "STAY_OUT":
            return "STAY_OUT"
        immediate_plan_id = _plan_id_for_scheduled_action(action, delay_laps=1)
        if immediate_plan_id in plans:
            return immediate_plan_id
        fallback_plan_ids = sorted(
            plan_id for plan_id in plans if plan_id.startswith(f"{action}_L+")
        )
        if fallback_plan_ids:
            return fallback_plan_ids[0]
        return "STAY_OUT"

    def _simulate_plan(
        self,
        *,
        state: RaceState,
        target_driver: str,
        plan_actions: tuple[dict[str, int | str], ...],
        scenarios: list[Scenario],
        seed: int,
    ) -> list[SimulationOutcome]:
        outcomes: list[SimulationOutcome] = []
        target_action_schedule = {
            int(action["at_lap"]): str(action["action"])
            for action in plan_actions
        }
        for scenario_index, scenario in enumerate(scenarios):
            rng = random.Random(seed + scenario_index * 10007 + sum(map(ord, target_driver)))
            sim_state = _clone_state(state)
            cumulative_race_times = {
                driver_id: float(car.gap_to_leader_ms or 0.0)
                for driver_id, car in sim_state.cars.items()
            }
            target_per_lap_times_ms: list[float] = []
            target_pace_component_totals_ms = {
                "base": 0.0,
                "degradation": 0.0,
                "track_status": 0.0,
                "traffic": 0.0,
                "weather": 0.0,
            }
            target_pit_loss_ms = 0.0
            has_pitted = {driver_id: False for driver_id in sim_state.cars}
            pace_cache: dict[tuple[Any, ...], Any] = {}
            pit_cache: dict[tuple[Any, ...], Any] = {}

            for step, status_name in enumerate(scenario.track_status_path, start=1):
                sim_state.lap = state.lap + step - 1
                sim_state.track_status = _track_status_from_name(status_name)
                sim_state.weather = dict(scenario.weather_path[step - 1])

                planned_actions: dict[str, str] = {}
                for driver_id, _car in sim_state.cars.items():
                    if driver_id == target_driver:
                        planned_actions[driver_id] = target_action_schedule.get(
                            state.lap + step,
                            "STAY_OUT",
                        )
                        continue
                    if has_pitted[driver_id]:
                        planned_actions[driver_id] = "STAY_OUT"
                        continue
                    planned_actions[driver_id] = self._sample_opponent_action(
                        state=sim_state,
                        driver_id=driver_id,
                        race_total_laps=_race_total_laps(
                            state=state,
                            fallback_horizon=len(scenario.track_status_path),
                        ),
                        pit_cache=pit_cache,
                        rng=rng,
                    )
                    if planned_actions[driver_id] != "STAY_OUT":
                        has_pitted[driver_id] = True

                lap_times: dict[str, float] = {}
                raw_pace_lap_times: dict[str, float] = {}
                for driver_id, current_action in planned_actions.items():
                    lap_time_ms, lap_diagnostics = self._simulate_driver_lap(
                        sim_state=sim_state,
                        driver_id=driver_id,
                        action=current_action,
                        pace_cache=pace_cache,
                    )
                    lap_times[driver_id] = lap_time_ms
                    raw_pace_lap_times[driver_id] = float(lap_diagnostics["pace_lap_time_ms"])
                    cumulative_race_times[driver_id] += lap_time_ms
                    if driver_id == target_driver:
                        target_per_lap_times_ms.append(
                            float(lap_diagnostics["pace_lap_time_ms"])
                        )
                        target_pit_loss_ms += float(lap_diagnostics["pit_loss_ms_used"])
                        for component, value in lap_diagnostics["pace_components_ms"].items():
                            target_pace_component_totals_ms[component] += float(value)
                    if current_action != "STAY_OUT" and driver_id == target_driver:
                        has_pitted[driver_id] = True

                self._update_running_order(
                    sim_state,
                    cumulative_race_times,
                    lap_times,
                    raw_pace_lap_times,
                )

            total_time_ms = accumulate_plan_total_time(
                target_per_lap_times_ms,
                pit_loss_ms=target_pit_loss_ms,
            )
            expected_total_time_ms = sum(target_per_lap_times_ms) + target_pit_loss_ms
            if not math.isclose(total_time_ms, expected_total_time_ms, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("target horizon total does not match per-lap accumulation")

            outcomes.append(
                SimulationOutcome(
                    total_time_ms=total_time_ms,
                    final_position=sim_state.cars[target_driver].position,
                    pit_loss_ms_used=target_pit_loss_ms,
                    per_lap_times_ms=target_per_lap_times_ms,
                    pace_component_totals_ms=target_pace_component_totals_ms,
                )
            )
        return outcomes

    def _sample_opponent_action(
        self,
        *,
        state: RaceState,
        driver_id: str,
        race_total_laps: int,
        pit_cache: dict[tuple[Any, ...], Any],
        rng: random.Random,
    ) -> str:
        rule_mask = action_mask(
            state,
            driver_id,
            state.lap,
            race_total_laps,
            deadline_laps=self.config.two_dry_deadline_laps,
        )
        key = _pit_cache_key(state, driver_id)
        if key not in pit_cache:
            pit_cache[key] = self.suite.pit_policy_model.predict_pit_prob(
                state,
                driver_id,
                window_laps=1,
            )
        prediction = pit_cache[key]
        if rng.random() >= prediction.p_pit_in_window:
            if rule_mask.get("STAY_OUT", True):
                return "STAY_OUT"
            return _first_feasible_action(rule_mask)
        sampled_action = _compound_distribution_to_action(prediction.p_compound)
        if rule_mask.get(sampled_action, True):
            return sampled_action
        if rule_mask.get("STAY_OUT", True):
            return "STAY_OUT"
        return _first_feasible_action(rule_mask)

    def _simulate_driver_lap(
        self,
        *,
        sim_state: RaceState,
        driver_id: str,
        action: str,
        pace_cache: dict[tuple[Any, ...], Any],
    ) -> tuple[float, dict[str, object]]:
        car = sim_state.cars[driver_id]
        pit_penalty = 0.0
        is_outlap = False
        if action != "STAY_OUT":
            car.tyre_compound = _tyre_compound_from_action(action)
            car.tyre_age_laps = 0
            car.stint_id += 1
            _record_compound_usage(car)
            pit_penalty = _pit_loss_ms(sim_state.track_status) + 2500.0
            is_outlap = True
        else:
            car.tyre_age_laps += 1

        car.pit_out = is_outlap
        car.pit_in = action != "STAY_OUT"
        car.is_pitting = action != "STAY_OUT"
        car.cleaning_flags = replace(
            car.cleaning_flags,
            is_inlap=action != "STAY_OUT",
            is_outlap=is_outlap,
            is_sc_vsc=sim_state.track_status in {TrackStatus.SC, TrackStatus.VSC},
        )
        cache_key = _pace_cache_key(sim_state, driver_id)
        if cache_key not in pace_cache:
            pace_cache[cache_key] = self.suite.pace_model.predict_lap_time(sim_state, driver_id)
        prediction = pace_cache[cache_key]
        lap_time = pit_penalty + prediction.mean_ms
        car.last_lap_time_ms = prediction.mean_ms
        if not car.cleaning_flags.is_sc_vsc and not car.cleaning_flags.is_outlap:
            recent = list(car.recent_lap_times_ms)
            recent.append(prediction.mean_ms)
            car.recent_lap_times_ms = recent[-5:]
        return lap_time, {
            "pace_lap_time_ms": prediction.mean_ms,
            "pit_loss_ms_used": pit_penalty,
            "pace_components_ms": dict(prediction.components),
        }

    def _update_running_order(
        self,
        sim_state: RaceState,
        cumulative_times: dict[str, float],
        lap_times: dict[str, float],
        raw_pace_lap_times: dict[str, float],
    ) -> None:
        ordered = sorted(cumulative_times.items(), key=lambda item: (item[1], item[0]))
        leader_time = ordered[0][1]
        previous_time: float | None = None
        for position, (driver_id, total_time) in enumerate(ordered, start=1):
            car = sim_state.cars[driver_id]
            car.position = position
            car.gap_to_leader_ms = total_time - leader_time
            car.interval_ahead_ms = 0.0 if previous_time is None else total_time - previous_time
            previous_time = total_time
            car.interval_behind_ms = None
            car.last_lap_time_ms = raw_pace_lap_times[driver_id]
            car.is_pitting = False
            car.pit_in = False

        for idx, (driver_id, total_time) in enumerate(ordered[:-1]):
            next_time = ordered[idx + 1][1]
            sim_state.cars[driver_id].interval_behind_ms = next_time - total_time

    def _plan_from_outcomes(
        self,
        *,
        state: RaceState,
        target_driver: str,
        plan_id: str,
        actions: list[dict[str, int | str]],
        outcomes: list[SimulationOutcome],
        baseline_outcomes: list[SimulationOutcome],
        seed: int,
    ) -> Plan:
        deltas = [
            compute_delta_time(outcome.total_time_ms, baseline.total_time_ms)
            for baseline, outcome in zip(baseline_outcomes, outcomes, strict=True)
        ]
        initial_position = state.cars[target_driver].position
        p_gain = sum(
            1 for outcome in outcomes if outcome.final_position <= initial_position - 1
        ) / len(outcomes)
        mean_delta = sum(deltas) / len(deltas)
        sigma = _distribution_sigma(deltas)
        plan_totals = [outcome.total_time_ms for outcome in outcomes]
        baseline_totals = [outcome.total_time_ms for outcome in baseline_outcomes]
        horizon_laps = len(outcomes[0].per_lap_times_ms) if outcomes else 0
        component_totals = _mean_component_totals(outcomes)
        diagnostics = _build_plan_diagnostics(
            plan_totals=plan_totals,
            baseline_totals=baseline_totals,
            outcomes=outcomes,
            baseline_outcomes=baseline_outcomes,
            deltas=deltas,
            horizon_laps=horizon_laps,
            n_scenarios=len(outcomes),
            seed_note=str(seed),
            delta_time_per_lap_cap_ms=self.config.sanity_delta_time_per_lap_cap_ms,
            p_gain=p_gain,
            risk_sigma_ms=sigma,
            component_totals=component_totals,
        )
        diagnostics["scheduled_pit_lap"] = (
            int(actions[0]["at_lap"]) if actions else None
        )
        diagnostics["scheduled_compound"] = _planned_compound_from_actions(actions)
        diagnostics["immediate_action"] = _immediate_action_for_plan(
            current_lap=state.lap,
            actions=actions,
        )
        return Plan(
            plan_id=plan_id,
            actions=actions,
            metrics=PlanMetrics(
                delta_time_mean_ms=mean_delta,
                delta_time_p10_ms=quantile(deltas, 0.10) or mean_delta,
                delta_time_p50_ms=quantile(deltas, 0.50) or mean_delta,
                delta_time_p90_ms=quantile(deltas, 0.90) or mean_delta,
                p_gain_pos_ge_1=p_gain,
                risk_sigma_ms=sigma,
            ),
            explanations=[],
            counterfactuals={},
            contributions=_build_plan_contributions(diagnostics),
            diagnostics=diagnostics,
            is_suspicious=bool(diagnostics["is_suspicious"]),
            suspicion_reason=(
                str(diagnostics["suspicion_reason"])
                if diagnostics["suspicion_reason"] is not None
                else None
            ),
        )

    @staticmethod
    def _rule_tyre_age_action(
        state: RaceState,
        target_driver: str,
        thresholds: dict[str, int],
    ) -> str:
        car = state.cars[target_driver]
        threshold = thresholds.get(car.tyre_compound.value, thresholds[TyreCompound.UNKNOWN.value])
        if car.tyre_age_laps >= threshold:
            return _pit_action_for_compound(car.tyre_compound)
        return "STAY_OUT"

    @staticmethod
    def _copy_action(state: RaceState, target_driver: str, copy_policy: str) -> str:
        reference = _reference_car(
            state=state,
            target_driver=target_driver,
            copy_policy=copy_policy,
        )
        if reference is None:
            return "STAY_OUT"
        if reference.pit_in or reference.pit_out:
            return _pit_action_for_compound(state.cars[target_driver].tyre_compound)
        return "STAY_OUT"


def validate_recommendation_bundle(
    bundle: RecommendationBundle,
    *,
    expected_top_k: int | None = None,
) -> None:
    if expected_top_k is not None and len(bundle.top_k) != expected_top_k and not bundle.warnings:
        raise ValueError("bundle top_k length does not match expectation")
    if not bundle.top_k:
        raise ValueError("bundle top_k must not be empty")
    for plan in bundle.top_k:
        if len(plan.explanations) < 2 and bundle.top_k[0].metrics.risk_sigma_ms >= 0:
            raise ValueError(f"plan {plan.plan_id} does not meet minimum explanation count")
        if plan.metrics.delta_time_p10_ms > plan.metrics.delta_time_p90_ms:
            raise ValueError(f"plan {plan.plan_id} has invalid quantiles")


def recommendation_bundle_to_dict(bundle: RecommendationBundle) -> dict[str, Any]:
    return {
        "session_id": bundle.session_id,
        "lap": bundle.lap,
        "target_driver": bundle.target_driver,
        "generated_at_ts": bundle.generated_at_ts,
        "top_k": [_plan_to_dict(plan) for plan in bundle.top_k],
        "baselines": {
            name: {
                "baseline_plan_id": comparison.baseline_plan_id,
                "delta_time_mean_ms": comparison.delta_time_mean_ms,
                "notes": comparison.notes,
            }
            for name, comparison in bundle.baselines.items()
        },
        "ground_truth": dict(bundle.ground_truth),
        "assumptions_hash": bundle.assumptions_hash,
        "model_versions": dict(bundle.model_versions),
        "warnings": list(bundle.warnings),
    }


def _plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "actions": list(plan.actions),
        "metrics": asdict(plan.metrics),
        "explanations": [asdict(explanation) for explanation in plan.explanations],
        "counterfactuals": dict(plan.counterfactuals),
        "contributions": dict(plan.contributions),
        "diagnostics": dict(plan.diagnostics),
        "is_suspicious": plan.is_suspicious,
        "suspicion_reason": plan.suspicion_reason,
    }


def _clone_state(state: RaceState) -> RaceState:
    return RaceState(
        session_id=state.session_id,
        lap=state.lap,
        track_status=state.track_status,
        total_laps=state.total_laps,
        weather=dict(state.weather),
        cars={
            driver_id: replace(
                car,
                recent_lap_times_ms=list(car.recent_lap_times_ms),
                used_dry_compounds=set(car.used_dry_compounds),
                cleaning_flags=replace(car.cleaning_flags),
            )
            for driver_id, car in state.cars.items()
        },
        warnings=list(state.warnings),
    )


def _pace_cache_key(state: RaceState, driver_id: str) -> tuple[Any, ...]:
    car = state.cars[driver_id]
    recent_anchor = (
        median(car.recent_lap_times_ms)
        if car.recent_lap_times_ms
        else car.last_lap_time_ms
    )
    rainfall = state.weather.get("rainfall")
    track_c = state.weather.get("track_c")
    return (
        driver_id,
        state.track_status.value,
        car.tyre_compound.value,
        car.tyre_age_laps,
        round(car.gap_to_leader_ms or 0.0, 1),
        round(car.interval_ahead_ms or 0.0, 1),
        round(car.interval_behind_ms or 0.0, 1),
        round(recent_anchor or 0.0, 1),
        round(rainfall or 0.0, 3),
        round(track_c or 0.0, 1),
        car.cleaning_flags.is_traffic_heavy,
        car.cleaning_flags.is_outlap,
    )


def _pit_cache_key(state: RaceState, driver_id: str) -> tuple[Any, ...]:
    car = state.cars[driver_id]
    return (
        driver_id,
        state.track_status.value,
        car.tyre_compound.value,
        car.tyre_age_laps,
        round(car.last_lap_time_ms or 0.0, 1),
        round(car.interval_ahead_ms or 0.0, 1),
        round(car.interval_behind_ms or 0.0, 1),
        car.cleaning_flags.is_traffic_heavy,
        tuple(sorted(car.used_dry_compounds)),
        car.used_wet,
    )


def _record_compound_usage(car: CarState) -> None:
    if car.tyre_compound in {TyreCompound.SOFT, TyreCompound.MEDIUM, TyreCompound.HARD}:
        car.used_dry_compounds.add(car.tyre_compound.value)
    elif car.tyre_compound in {TyreCompound.INTER, TyreCompound.WET}:
        car.used_wet = True


def _race_total_laps(*, state: RaceState, fallback_horizon: int) -> int:
    return state.total_laps or (state.lap + fallback_horizon)


def _first_feasible_action(mask: dict[str, bool]) -> str:
    for action in ("PIT_TO_MEDIUM", "PIT_TO_HARD", "PIT_TO_SOFT", "PIT_TO_INTER", "PIT_TO_WET"):
        if mask.get(action, False):
            return action
    return "STAY_OUT"


def _track_status_from_name(name: str) -> TrackStatus:
    normalized = name.strip().upper()
    for status in TrackStatus:
        if status.value == normalized:
            return status
    return TrackStatus.GREEN


def _pit_loss_ms(track_status: TrackStatus) -> float:
    return 12000.0 if track_status in {TrackStatus.SC, TrackStatus.VSC} else 22000.0


def _distribution_sigma(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _plan_comparison(best_plan: Plan, baseline_plan: Plan) -> PlanComparison:
    best_total = float(best_plan.diagnostics.get("plan_total_time_ms", 0.0))
    baseline_total = float(baseline_plan.diagnostics.get("plan_total_time_ms", 0.0))
    return PlanComparison(
        baseline_plan_id=baseline_plan.plan_id,
        delta_time_mean_ms=compute_delta_time(best_total, baseline_total),
        notes="Positive means the selected plan is estimated to outperform the baseline.",
    )


def _plan_actions(
    current_lap: int,
    action: str,
    *,
    delay_laps: int = 1,
) -> list[dict[str, int | str]]:
    if action == "STAY_OUT":
        return []
    return [{"at_lap": current_lap + delay_laps, "action": action}]


def _plan_id_for_scheduled_action(action: str, delay_laps: int) -> str:
    if action == "STAY_OUT":
        return "STAY_OUT"
    return f"{action}_L+{delay_laps}"


def _plan_respects_rule_deadline(
    *,
    state: RaceState,
    driver_id: str,
    plan: Plan,
    race_total_laps: int,
    deadline_laps: int,
    planned_lap: int | None = None,
) -> bool:
    car = state.cars[driver_id]
    if car.used_wet or len(car.used_dry_compounds) >= 2:
        return True

    last_safe_pit_lap = max(state.lap + 1, race_total_laps - deadline_laps)
    if planned_lap is None:
        planned_lap = min(
            (
                int(action["at_lap"])
                for action in plan.actions
                if str(action["action"]).startswith("PIT_TO_")
            ),
            default=None,
        )
    if planned_lap is None:
        return (race_total_laps - state.lap) > deadline_laps

    planned_compound = _planned_compound_from_actions(plan.actions)
    if planned_compound is None:
        return (race_total_laps - state.lap) > deadline_laps
    if planned_compound in {TyreCompound.INTER.value, TyreCompound.WET.value}:
        return True
    if planned_compound in car.used_dry_compounds:
        return False
    return planned_lap <= last_safe_pit_lap


def _reference_car(*, state: RaceState, target_driver: str, copy_policy: str) -> CarState | None:
    ordered = sorted(state.cars.values(), key=lambda car: (car.position, car.driver_id))
    if copy_policy == "leader":
        return ordered[0] if ordered and ordered[0].driver_id != target_driver else None
    target = state.cars[target_driver]
    for car in ordered:
        if car.position == target.position - 1:
            return car
    return ordered[0] if ordered and ordered[0].driver_id != target_driver else None


def _compound_distribution_to_action(distribution: dict[str, float]) -> str:
    if not distribution:
        return "PIT_TO_HARD"
    compound = max(sorted(distribution), key=lambda key: distribution[key])
    return f"PIT_TO_{compound}"


def _pit_action_for_compound(compound: TyreCompound) -> str:
    if compound is TyreCompound.SOFT:
        return "PIT_TO_MEDIUM"
    if compound is TyreCompound.MEDIUM:
        return "PIT_TO_HARD"
    if compound is TyreCompound.HARD:
        return "PIT_TO_MEDIUM"
    if compound is TyreCompound.INTER:
        return "PIT_TO_WET"
    if compound is TyreCompound.WET:
        return "PIT_TO_INTER"
    return "PIT_TO_HARD"


def _tyre_compound_from_action(action: str) -> TyreCompound:
    suffix = action.removeprefix("PIT_TO_")
    for compound in TyreCompound:
        if compound.value == suffix:
            return compound
    return TyreCompound.UNKNOWN


def _planned_compound_from_actions(actions: list[dict[str, int | str]]) -> str | None:
    for action in actions:
        action_name = str(action["action"])
        if action_name.startswith("PIT_TO_"):
            return action_name.removeprefix("PIT_TO_")
    return None


def _immediate_action_for_plan(
    *,
    current_lap: int,
    actions: list[dict[str, int | str]],
) -> str:
    for action in actions:
        if int(action["at_lap"]) == current_lap + 1:
            return str(action["action"])
    return "STAY_OUT"


def _mean_component_totals(outcomes: list[SimulationOutcome]) -> dict[str, float]:
    if not outcomes:
        return {}
    totals: dict[str, float] = {}
    for outcome in outcomes:
        for component, value in outcome.pace_component_totals_ms.items():
            totals[component] = totals.get(component, 0.0) + value
    return {
        component: total / len(outcomes)
        for component, total in totals.items()
    }


def _build_plan_diagnostics(
    *,
    plan_totals: list[float],
    baseline_totals: list[float],
    outcomes: list[SimulationOutcome],
    baseline_outcomes: list[SimulationOutcome],
    deltas: list[float],
    horizon_laps: int,
    n_scenarios: int,
    seed_note: str,
    delta_time_per_lap_cap_ms: float,
    p_gain: float,
    risk_sigma_ms: float,
    component_totals: dict[str, float],
) -> dict[str, object]:
    plan_total_time_ms = sum(plan_totals) / len(plan_totals)
    baseline_total_time_ms = sum(baseline_totals) / len(baseline_totals)
    pit_loss_ms_used = sum(outcome.pit_loss_ms_used for outcome in outcomes) / len(outcomes)
    baseline_pit_loss_ms = sum(
        outcome.pit_loss_ms_used for outcome in baseline_outcomes
    ) / len(baseline_outcomes)
    baseline_component_totals = _mean_component_totals(baseline_outcomes)
    per_lap_component_summary = {
        f"{component}_mean_ms": total / horizon_laps if horizon_laps else 0.0
        for component, total in component_totals.items()
    }
    suspicion_reason = suspicion_reason_for_delta_time(
        delta_time_ms=sum(deltas) / len(deltas),
        horizon_laps=max(horizon_laps, 1),
        delta_time_per_lap_cap_ms=delta_time_per_lap_cap_ms,
    )
    return {
        "horizon_laps": horizon_laps,
        "n_scenarios": n_scenarios,
        "seed": seed_note,
        "pit_loss_ms_used": pit_loss_ms_used,
        "baseline_pit_loss_ms": baseline_pit_loss_ms,
        "plan_component_totals_ms": dict(component_totals),
        "baseline_component_totals_ms": dict(baseline_component_totals),
        "per_lap_time_components_summary_ms": per_lap_component_summary,
        "baseline_total_time_ms": baseline_total_time_ms,
        "plan_total_time_ms": plan_total_time_ms,
        "plan_total_time_p10_ms": quantile(plan_totals, 0.10) or plan_total_time_ms,
        "plan_total_time_p50_ms": quantile(plan_totals, 0.50) or plan_total_time_ms,
        "plan_total_time_p90_ms": quantile(plan_totals, 0.90) or plan_total_time_ms,
        "baseline_total_time_p50_ms": quantile(baseline_totals, 0.50) or baseline_total_time_ms,
        "delta_time_cap_ms": delta_time_cap_ms(
            horizon_laps=max(horizon_laps, 1),
            delta_time_per_lap_cap_ms=delta_time_per_lap_cap_ms,
        ),
        "is_suspicious": suspicion_reason is not None,
        "suspicion_reason": suspicion_reason,
        "p_gain_pos_ge_1": p_gain,
        "risk_sigma_ms": risk_sigma_ms,
    }


def _build_plan_contributions(diagnostics: dict[str, object]) -> dict[str, float]:
    return contribution_breakdown(
        plan_pit_loss_ms=float(diagnostics.get("pit_loss_ms_used", 0.0)),
        baseline_pit_loss_ms=float(diagnostics.get("baseline_pit_loss_ms", 0.0)),
        plan_components_ms=dict(diagnostics.get("plan_component_totals_ms", {})),
        baseline_components_ms=dict(diagnostics.get("baseline_component_totals_ms", {})),
    )
