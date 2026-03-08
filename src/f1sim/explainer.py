"""Reason-code explanations and counterfactuals for recommendation plans."""

from __future__ import annotations

from typing import Any

from f1sim.contracts import Explanation, Plan
from f1sim.rules import compounds_used_count, is_two_dry_required
from f1sim.state import RaceState, TrackStatus


def build_plan_explanations(
    *,
    state: RaceState,
    target_driver: str,
    plan: Plan,
    plans_by_id: dict[str, Plan],
    degradation_model: Any,
    pit_policy_model: Any,
    n_scenarios: int,
    race_total_laps: int,
    deadline_laps: int,
) -> list[Explanation]:
    car = state.cars[target_driver]
    action = _plan_action(plan)
    explanations: list[Explanation] = []

    degradation = degradation_model.predict_delta(
        car.tyre_compound.value,
        car.tyre_age_laps,
        {"track_status": state.track_status.value, "weather": state.weather},
    )
    pit_prob = pit_policy_model.predict_pit_prob(state, target_driver, window_laps=3)
    laps_remaining = max(0, race_total_laps - state.lap)

    if state.track_status in {TrackStatus.SC, TrackStatus.VSC}:
        explanations.append(
            Explanation(
                code="SC_WINDOW",
                text="Neutralized conditions shrink the effective pit-loss window.",
                evidence={
                    "track_status": state.track_status.value,
                    "n_scenarios": n_scenarios,
                    "pit_loss_ms": 12000.0,
                },
            )
        )

    reference = _car_ahead(state, target_driver)
    reference_gap_ms = 0.0
    if reference is not None:
        reference_gap_ms = reference.interval_behind_ms or car.interval_ahead_ms or 0.0
    if reference is not None and reference_gap_ms <= 3000.0:
        explanations.append(
            Explanation(
                code="UNDERCUT_THREAT",
                text="A nearby car ahead makes an undercut or cover reaction relevant.",
                evidence={
                    "reference_driver": reference.driver_id,
                    "gap_ms": car.interval_ahead_ms,
                    "reference_position": reference.position,
                },
            )
        )

    if car.cleaning_flags.is_traffic_heavy:
        explanations.append(
            Explanation(
                code="TRAFFIC_PENALTY",
                text="Traffic-heavy running is increasing expected lap-time loss.",
                evidence={
                    "interval_ahead_ms": car.interval_ahead_ms,
                    "interval_behind_ms": car.interval_behind_ms,
                },
            )
        )

    explanations.append(
        Explanation(
            code="TRACK_POSITION",
            text="Track position and current gaps frame the near-term action value.",
            evidence={
                "position": car.position,
                "gap_to_leader_ms": car.gap_to_leader_ms,
                "interval_ahead_ms": car.interval_ahead_ms,
                "action": action,
            },
        )
    )
    explanations.append(
        Explanation(
            code="TYRE_CLIFF",
            text="Tyre age and heuristic degradation create a cliff-risk proxy.",
            evidence={
                "compound": car.tyre_compound.value,
                "tyre_age": car.tyre_age_laps,
                "cliff_risk": degradation.cliff_risk,
                "delta_mean_ms": degradation.delta_mean_ms,
                "p_pit_next_3": pit_prob.p_pit_in_window,
            },
        )
    )
    planned_new_compound = _planned_new_compound(plan)
    if (
        is_two_dry_required(state, target_driver)
        and compounds_used_count(state, target_driver) < 2
        and action != "STAY_OUT"
        and planned_new_compound is not None
        and laps_remaining <= deadline_laps
    ):
        explanations.append(
            Explanation(
                code="RULE_COMPLIANCE",
                text="The pit plan adds a legal compound path before the dry-tyre deadline.",
                evidence={
                    "used_wet": car.used_wet,
                    "used_dry_compounds": sorted(car.used_dry_compounds),
                    "required_distinct_dry": 2,
                    "laps_remaining": laps_remaining,
                    "deadline_laps": deadline_laps,
                    "planned_new_compound": planned_new_compound,
                },
            )
        )

    rainfall = state.weather.get("rainfall")
    if rainfall is not None and rainfall > 0.0:
        explanations.append(
            Explanation(
                code="RAIN_RISK",
                text="Weather inputs indicate rain-affected rollout risk.",
                evidence={
                    "rainfall": rainfall,
                    "track_c": state.weather.get("track_c"),
                },
            )
        )

    unique: list[Explanation] = []
    seen_codes: set[str] = set()
    for explanation in explanations:
        if explanation.code not in seen_codes:
            unique.append(explanation)
            seen_codes.add(explanation.code)
    if len(unique) <= 4:
        return unique
    compliance = [explanation for explanation in unique if explanation.code == "RULE_COMPLIANCE"]
    if not compliance:
        return unique[:4]
    trimmed = compliance + [
        explanation for explanation in unique if explanation.code != "RULE_COMPLIANCE"
    ]
    return trimmed[:4]


def build_plan_counterfactuals(
    *,
    plan: Plan,
    plans_by_id: dict[str, Plan],
) -> dict[str, dict[str, float | str]]:
    counterfactuals = {
        "vs_STAY_OUT": _compare_plans(plan, plans_by_id["STAY_OUT"]),
    }
    pit_next_lap_plan = _best_pit_next_lap_plan(plans_by_id)
    if pit_next_lap_plan is not None:
        counterfactuals["vs_PIT_NEXT_LAP"] = _compare_plans(plan, pit_next_lap_plan)
    return counterfactuals


def _compare_plans(plan: Plan, reference: Plan) -> dict[str, float | str]:
    return {
        "reference_plan_id": reference.plan_id,
        "reference_action": _plan_action(reference),
        "delta_time_mean_ms": (
            plan.metrics.delta_time_mean_ms - reference.metrics.delta_time_mean_ms
        ),
        "delta_time_p50_ms": plan.metrics.delta_time_p50_ms - reference.metrics.delta_time_p50_ms,
    }


def _best_pit_next_lap_plan(plans_by_id: dict[str, Plan]) -> Plan | None:
    pit_plans = [plan for plan in plans_by_id.values() if _plan_action(plan) != "STAY_OUT"]
    if not pit_plans:
        return None
    return max(pit_plans, key=lambda plan: plan.metrics.delta_time_mean_ms)


def _plan_action(plan: Plan) -> str:
    if not plan.actions:
        return "STAY_OUT"
    return str(plan.actions[0]["action"])


def _car_ahead(state: RaceState, target_driver: str) -> Any:
    target = state.cars[target_driver]
    for car in state.cars.values():
        if car.position == target.position - 1:
            return car
    return None


def _planned_new_compound(plan: Plan) -> str | None:
    action = _plan_action(plan)
    if not action.startswith("PIT_TO_"):
        return None
    return action.removeprefix("PIT_TO_")
