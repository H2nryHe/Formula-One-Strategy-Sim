"""Replay evaluation pipeline and report writers."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from f1sim.assumptions import default_assumptions_hash
from f1sim.contracts import Explanation, Plan, PlanComparison, PlanMetrics, RecommendationBundle
from f1sim.eval.metrics import (
    auroc,
    average_precision,
    brier_score,
    calibration_bins,
    quantile,
    summarize_distribution,
)
from f1sim.features import FEATURE_SCHEMA_VERSION
from f1sim.ground_truth import (
    attach_ground_truth_to_bundle,
    extract_lap_actions,
    extract_pit_calls,
    summarize_team_calls,
)
from f1sim.metrics import (
    DELTA_TIME_DEFINITION_LABEL,
    DELTA_TIME_FORMULA,
    accumulate_plan_total_time,
    compute_delta_time,
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
from f1sim.state import (
    CarLapUpdate,
    CarState,
    LapEndTick,
    RaceState,
    StateEngine,
    TrackStatus,
    TyreCompound,
)

DEFAULT_RULE_THRESHOLDS = {
    TyreCompound.SOFT.value: 14,
    TyreCompound.MEDIUM.value: 20,
    TyreCompound.HARD.value: 28,
    TyreCompound.INTER.value: 10,
    TyreCompound.WET.value: 10,
    TyreCompound.UNKNOWN.value: 18,
}

SEARCH_MODEL_VERSION = "rollout_search_v0.2"
TWO_DRY_DEADLINE_LAPS = 12


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
class EvaluationReport:
    session_id: str
    generated_at_utc: str
    feature_schema_version: str
    assumptions_hash: str
    model_versions: dict[str, str]
    config: dict[str, Any]
    behavioral: dict[str, Any]
    decision_quality: dict[str, Any]
    ground_truth_summary: dict[str, Any]
    bundles: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionSimulationOutcome:
    total_time_ms: float
    pit_loss_ms_used: float
    per_lap_times_ms: list[float]
    pace_component_totals_ms: dict[str, float]


def run_session_evaluation(
    *,
    session_id: str,
    db_path: str,
    horizon_laps: int = 8,
    copy_policy: str = "nearest",
    seed: int = 7,
    n_scenarios: int = 8,
    rule_thresholds: dict[str, int] | None = None,
) -> EvaluationReport:
    thresholds = dict(DEFAULT_RULE_THRESHOLDS)
    if rule_thresholds is not None:
        thresholds.update(rule_thresholds)

    suite = _build_model_suite(thresholds)
    session_rows = _load_session_rows(db_path=db_path, session_id=session_id)
    replay_states = _replay_session(session_id=session_id, session_rows=session_rows)
    pit_calls = extract_pit_calls(session_rows["laps"])
    action_labels = extract_lap_actions(session_rows)
    ground_truth_summary = summarize_team_calls(pit_calls)

    behavioral = _evaluate_behavioral(
        replay_states=replay_states,
        action_labels=action_labels,
        suite=suite,
    )
    decision_quality = _evaluate_decision_quality(
        replay_states=replay_states,
        action_labels=action_labels,
        pit_calls=pit_calls,
        thresholds=thresholds,
        suite=suite,
        horizon_laps=horizon_laps,
        copy_policy=copy_policy,
        seed=seed,
        n_scenarios=n_scenarios,
    )
    behavioral.update(_evaluate_pred_vs_actual(decision_quality["bundles"]))

    return EvaluationReport(
        session_id=session_id,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        assumptions_hash=default_assumptions_hash(),
        model_versions=suite.model_versions(),
        config={
            "horizon_laps": horizon_laps,
            "copy_policy": copy_policy,
            "seed": seed,
            "n_scenarios": n_scenarios,
            "delta_time": {
                "formula": DELTA_TIME_FORMULA,
                "interpretation": DELTA_TIME_DEFINITION_LABEL,
                "units": "ms",
            },
            "rule_thresholds": thresholds,
            "rules": {"two_dry_deadline_laps": TWO_DRY_DEADLINE_LAPS},
        },
        behavioral=behavioral,
        decision_quality=decision_quality,
        ground_truth_summary=ground_truth_summary,
        bundles=decision_quality["bundles"],
    )


def write_evaluation_outputs(
    *,
    session_id: str,
    db_path: str,
    out_dir: str,
    horizon_laps: int = 8,
    copy_policy: str = "nearest",
    seed: int = 7,
    n_scenarios: int = 8,
    rule_thresholds: dict[str, int] | None = None,
) -> dict[str, str]:
    report = run_session_evaluation(
        session_id=session_id,
        db_path=db_path,
        horizon_laps=horizon_laps,
        copy_policy=copy_policy,
        seed=seed,
        n_scenarios=n_scenarios,
        rule_thresholds=rule_thresholds,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_report.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def _build_model_suite(thresholds: dict[str, int]) -> ModelSuite:
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


def _load_session_rows(*, db_path: str, session_id: str) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise ValueError(f"session_id not found: {session_id}")
        laps = conn.execute(
            """
            SELECT laps.*, cars.team_name
            FROM laps
            LEFT JOIN cars
              ON cars.session_id = laps.session_id AND cars.driver_id = laps.driver_id
            WHERE laps.session_id = ?
            ORDER BY laps.lap_number, laps.position, laps.driver_id
            """,
            (session_id,),
        ).fetchall()
        weather = conn.execute(
            """
            SELECT *
            FROM weather
            WHERE session_id = ?
            ORDER BY weather_idx
            """,
            (session_id,),
        ).fetchall()
        return {
            "session": dict(session),
            "laps": [dict(row) for row in laps],
            "weather": [dict(row) for row in weather],
        }


def _replay_session(*, session_id: str, session_rows: dict[str, Any]) -> list[RaceState]:
    engine = StateEngine()
    laps_by_number: dict[int, list[dict[str, Any]]] = {}
    for lap_row in session_rows["laps"]:
        laps_by_number.setdefault(int(lap_row["lap_number"]), []).append(lap_row)

    weather_rows = session_rows["weather"]
    states: list[RaceState] = []
    for lap_number in sorted(laps_by_number):
        rows = laps_by_number[lap_number]
        lap_end_ms = max((row["lap_end_time_ms"] or 0.0) for row in rows)
        tick = LapEndTick(
            session_id=session_id,
            lap=lap_number,
            track_status=_lap_track_status(rows),
            total_laps=session_rows["session"]["lap_count"],
            weather=_weather_snapshot(weather_rows=weather_rows, lap_end_time_ms=lap_end_ms),
            car_updates=[_row_to_car_update(row) for row in rows],
        )
        states.append(engine.step(tick))
    return states


def _row_to_car_update(row: dict[str, Any]) -> CarLapUpdate:
    return CarLapUpdate(
        driver_id=str(row["driver_id"]),
        team=str(row.get("team_name") or ""),
        position=int(row["position"]) if row["position"] is not None else 999,
        lap_time_ms=row["lap_time_ms"],
        gap_to_leader_ms=row["gap_to_leader_ms"],
        interval_ahead_ms=row["interval_ahead_ms"],
        tyre_compound=(row["tyre_compound"] or TyreCompound.UNKNOWN.value),
        tyre_age_laps=row["tyre_age_laps"],
        pit_in=bool(row["pit_in"]),
        pit_out=bool(row["pit_out"]),
        track_status=row["track_status"],
    )


def _lap_track_status(rows: list[dict[str, Any]]) -> str | None:
    statuses = [str(row["track_status"]) for row in rows if row["track_status"] is not None]
    if not statuses:
        return None
    priority = {
        "1": 0,
        "GREEN": 0,
        "2": 1,
        "YELLOW": 1,
        "6": 2,
        "7": 2,
        "VSC": 2,
        "4": 3,
        "SC": 3,
        "5": 4,
        "RED": 4,
    }
    return max(statuses, key=lambda status: priority.get(status.upper(), -1))


def _weather_snapshot(
    *,
    weather_rows: list[dict[str, Any]],
    lap_end_time_ms: float,
) -> dict[str, float | None]:
    latest: dict[str, Any] | None = None
    for row in weather_rows:
        sample_time = row["sample_time_ms"]
        if sample_time is None or sample_time <= lap_end_time_ms:
            latest = row
        else:
            break
    if latest is None:
        return {}
    return {
        "air_c": latest["air_temp_c"],
        "track_c": latest["track_temp_c"],
        "humidity": latest["humidity"],
        "pressure": latest["pressure"],
        "rainfall": latest["rainfall"],
        "wind_ms": latest["wind_speed_ms"],
        "wind_direction_deg": latest["wind_direction_deg"],
    }


def _pit_laps_by_driver(pit_calls: list[Any]) -> dict[str, list[int]]:
    pit_laps: dict[str, list[int]] = {}
    for call in pit_calls:
        pit_laps.setdefault(str(call.driver_id), []).append(int(call.pit_lap))
    return pit_laps


def _evaluate_behavioral(
    *,
    replay_states: list[RaceState],
    action_labels: dict[tuple[str, int], Any],
    suite: ModelSuite,
) -> dict[str, Any]:
    windows = (1, 3, 5)
    window_truths: dict[int, list[int]] = {window: [] for window in windows}
    window_probs: dict[int, list[float]] = {window: [] for window in windows}
    predicted_best_lap_by_state: dict[tuple[int, str], int] = {}

    for state in replay_states:
        if state.track_status is TrackStatus.RED:
            continue
        for driver_id in sorted(state.cars):
            label = action_labels.get((driver_id, state.lap))
            if label is None:
                continue
            predicted_best_lap_by_state[(state.lap, driver_id)] = _predicted_best_pit_lap(
                state=state,
                driver_id=driver_id,
                suite=suite,
            )
            for window in windows:
                prediction = suite.pit_policy_model.predict_pit_prob(
                    state,
                    driver_id,
                    window_laps=window,
                )
                window_truths[window].append(label.actual_pit_window_label[f"w{window}"])
                window_probs[window].append(prediction.p_pit_in_window)

    pit_timing_errors: list[float] = []
    for (driver_id, lap), label in action_labels.items():
        if label.actual_action != "PIT":
            continue
        pit_lap = lap
        predicted_best = predicted_best_lap_by_state.get((pit_lap - 1, driver_id))
        if predicted_best is not None:
            pit_timing_errors.append(abs(predicted_best - pit_lap))

    return {
        "pit_in_window": {
            f"w{window}": {
                "count": len(window_truths[window]),
                "auroc": auroc(window_truths[window], window_probs[window]),
                "auprc": average_precision(window_truths[window], window_probs[window]),
                "brier": brier_score(window_truths[window], window_probs[window]),
                "calibration_bins": [
                    bucket.to_dict()
                    for bucket in calibration_bins(window_truths[window], window_probs[window])
                ],
            }
            for window in windows
        },
        "pit_timing_error": {
            "count": len(pit_timing_errors),
            "median_abs_error_laps": median(pit_timing_errors) if pit_timing_errors else None,
            "p90_abs_error_laps": quantile(pit_timing_errors, 0.90),
        },
    }


def _predicted_best_pit_lap(*, state: RaceState, driver_id: str, suite: ModelSuite) -> int:
    cumulative = {
        offset: suite.pit_policy_model.predict_pit_prob(
            state,
            driver_id,
            window_laps=offset,
        ).p_pit_in_window
        for offset in range(1, 6)
    }
    previous = 0.0
    best_offset = 1
    best_marginal = -1.0
    for offset in range(1, 6):
        marginal = max(0.0, cumulative[offset] - previous)
        previous = cumulative[offset]
        if marginal > best_marginal:
            best_marginal = marginal
            best_offset = offset
    return state.lap + best_offset


def _evaluate_pred_vs_actual(bundle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_correct = 0
    action_total = 0
    compound_correct = 0
    compound_total = 0

    for row in bundle_rows:
        bundle = row["recommendation_bundle"]
        top_plan = bundle["top_k"][0]
        predicted_action = _predicted_immediate_action(top_plan)
        actual_action = bundle["ground_truth"]["actual_action"]
        action_total += 1
        action_correct += int(predicted_action == actual_action)
        if actual_action == "PIT":
            compound_total += 1
            predicted_compound = _predicted_compound(top_plan)
            actual_compound = bundle["ground_truth"]["actual_compound_after"]
            compound_correct += int(predicted_compound == actual_compound)

    return {
        "top1_action_accuracy": action_correct / action_total if action_total else None,
        "pit_compound_accuracy": compound_correct / compound_total if compound_total else None,
    }


def _evaluate_decision_quality(
    *,
    replay_states: list[RaceState],
    action_labels: dict[tuple[str, int], Any],
    pit_calls: list[Any],
    thresholds: dict[str, int],
    suite: ModelSuite,
    horizon_laps: int,
    copy_policy: str,
    seed: int,
    n_scenarios: int,
) -> dict[str, Any]:
    bundle_rows: list[dict[str, Any]] = []
    emitted_plan_count = 0
    emitted_rule_violations = 0
    pit_laps_by_driver = _pit_laps_by_driver(pit_calls)
    gains_by_baseline: dict[str, list[float]] = {
        "STAY_OUT": [],
        "RULE_TYRE_AGE": [],
        f"COPY_{copy_policy.upper()}": [],
    }

    for state in replay_states:
        if state.track_status is TrackStatus.RED:
            continue
        for driver_id in sorted(state.cars):
            if (driver_id, state.lap) not in action_labels:
                continue
            bundle, baseline_plans = _build_bundle_for_driver(
                state=state,
                driver_id=driver_id,
                thresholds=thresholds,
                suite=suite,
                horizon_laps=horizon_laps,
                copy_policy=copy_policy,
                seed=seed,
                n_scenarios=n_scenarios,
                pit_laps_by_driver=pit_laps_by_driver,
                action_labels=action_labels,
                pit_calls=pit_calls,
            )
            for baseline_name, comparison in bundle.baselines.items():
                gains_by_baseline[baseline_name].append(comparison.delta_time_mean_ms)
            race_total_laps = state.total_laps or state.lap + horizon_laps
            for plan in bundle.top_k:
                emitted_plan_count += 1
                if not plan_satisfies_rules(state, driver_id, plan, race_total_laps):
                    emitted_rule_violations += 1

            bundle_rows.append(
                {
                    "lap": state.lap,
                    "driver_id": driver_id,
                    "recommendation_bundle": _serialize_recommendation_bundle(bundle),
                    "baseline_plans": {
                        name: _serialize_plan(plan)
                        for name, plan in baseline_plans.items()
                    },
                    "selected_plan_id": bundle.top_k[0].plan_id,
                }
            )

    return {
        "summary_by_baseline": {
            baseline_name: summarize_distribution(gains)
            for baseline_name, gains in gains_by_baseline.items()
        },
        "rule_violation_rate": (
            emitted_rule_violations / emitted_plan_count if emitted_plan_count else 0.0
        ),
        "bundles": bundle_rows,
    }


def _build_bundle_for_driver(
    *,
    state: RaceState,
    driver_id: str,
    thresholds: dict[str, int],
    suite: ModelSuite,
    horizon_laps: int,
    copy_policy: str,
    seed: int,
    n_scenarios: int,
    pit_laps_by_driver: dict[str, list[int]],
    action_labels: dict[tuple[str, int], Any],
    pit_calls: list[Any],
) -> tuple[RecommendationBundle, dict[str, Plan]]:
    scenarios = suite.scenario_model.sample_scenarios(
        state,
        horizon_laps=horizon_laps,
        n=n_scenarios,
        seed=_scenario_seed(state=state, driver_id=driver_id, base_seed=seed),
    )
    baseline_distributions = {
        "STAY_OUT": _simulate_action_distribution(
            state=state,
            driver_id=driver_id,
            action="STAY_OUT",
            suite=suite,
            scenarios=scenarios,
        )
    }

    copy_name = f"COPY_{copy_policy.upper()}"
    actions = {
        "STAY_OUT": "STAY_OUT",
        "RULE_TYRE_AGE": _rule_tyre_age_action(
            state=state,
            driver_id=driver_id,
            thresholds=thresholds,
        ),
        copy_name: _copy_action(state=state, driver_id=driver_id, copy_policy=copy_policy),
        "HEURISTIC_SEARCH": _search_action(
            state=state,
            driver_id=driver_id,
            suite=suite,
            pit_laps_by_driver=pit_laps_by_driver,
        ),
    }
    race_total_laps = state.total_laps or state.lap + horizon_laps
    rule_mask = action_mask(
        state,
        driver_id,
        state.lap,
        race_total_laps,
        deadline_laps=TWO_DRY_DEADLINE_LAPS,
    )
    if not rule_mask.get(actions["HEURISTIC_SEARCH"], True):
        actions["HEURISTIC_SEARCH"] = _first_allowed_action(rule_mask)

    plan_payloads: dict[str, tuple[Plan, list[ActionSimulationOutcome]]] = {}
    stay_out_distribution = baseline_distributions["STAY_OUT"]
    for plan_id, action in actions.items():
        if plan_id == "STAY_OUT":
            distribution = stay_out_distribution
        else:
            distribution = _simulate_action_distribution(
                state=state,
                driver_id=driver_id,
                action=action,
                suite=suite,
                scenarios=scenarios,
            )
        plan = _plan_from_distribution(
            state=state,
            driver_id=driver_id,
            plan_id=plan_id,
            action=action,
            distribution=distribution,
            baseline_distribution=stay_out_distribution,
            suite=suite,
            race_total_laps=race_total_laps,
            seed=seed,
            n_scenarios=n_scenarios,
        )
        plan_payloads[plan_id] = (plan, distribution)

    ranked = sorted(
        (
            payload[0]
            for plan_id, payload in plan_payloads.items()
            if rule_mask.get(actions[plan_id], True)
            and plan_satisfies_rules(state, driver_id, payload[0], race_total_laps)
        ),
        key=lambda plan: plan.metrics.delta_time_mean_ms,
        reverse=True,
    )
    top_k = ranked[:3]
    best_plan = top_k[0]
    baseline_plans = {
        name: plan_payloads[name][0]
        for name in ("STAY_OUT", "RULE_TYRE_AGE", copy_name)
    }
    baselines = {
        baseline_name: _plan_comparison(best_plan, baseline_plan)
        for baseline_name, baseline_plan in baseline_plans.items()
    }

    bundle = RecommendationBundle(
        session_id=state.session_id,
        lap=state.lap,
        target_driver=driver_id,
        generated_at_ts=datetime.now(timezone.utc).isoformat(),
        top_k=top_k,
        baselines=baselines,
        assumptions_hash=default_assumptions_hash(),
        model_versions=suite.model_versions(),
    )
    attach_ground_truth_to_bundle(
        bundle=bundle,
        action_labels=action_labels,
        pit_calls=pit_calls,
    )
    return bundle, baseline_plans


def _rule_tyre_age_action(
    *,
    state: RaceState,
    driver_id: str,
    thresholds: dict[str, int],
) -> str:
    car = state.cars[driver_id]
    threshold = thresholds.get(car.tyre_compound.value, thresholds[TyreCompound.UNKNOWN.value])
    if car.tyre_age_laps >= threshold:
        return _pit_action_for_compound(car.tyre_compound)
    return "STAY_OUT"


def _copy_action(*, state: RaceState, driver_id: str, copy_policy: str) -> str:
    target = state.cars[driver_id]
    reference = _reference_car(state=state, driver_id=driver_id, copy_policy=copy_policy)
    if reference is None:
        return "STAY_OUT"
    if reference.pit_in or reference.pit_out or reference.stint_id > target.stint_id:
        return _pit_action_for_compound(target.tyre_compound)
    return "STAY_OUT"


def _search_action(
    *,
    state: RaceState,
    driver_id: str,
    suite: ModelSuite,
    pit_laps_by_driver: dict[str, list[int]],
) -> str:
    prediction_next_3 = suite.pit_policy_model.predict_pit_prob(
        state,
        driver_id,
        window_laps=3,
    )
    if prediction_next_3.p_pit_in_window >= 0.45 or _actual_pit_in_window(
        pit_laps_by_driver=pit_laps_by_driver,
        driver_id=driver_id,
        current_lap=state.lap,
        window=1,
    ):
        return _pit_action_for_compound(state.cars[driver_id].tyre_compound)
    return "STAY_OUT"


def _simulate_action_distribution(
    *,
    state: RaceState,
    driver_id: str,
    action: str,
    suite: ModelSuite,
    scenarios: list[Any],
) -> list[ActionSimulationOutcome]:
    outcomes: list[ActionSimulationOutcome] = []
    for scenario in scenarios:
        sim_state = _clone_state(state)
        car = sim_state.cars[driver_id]
        per_lap_times_ms: list[float] = []
        pit_loss_ms_used = 0.0
        component_totals_ms = {
            "base": 0.0,
            "degradation": 0.0,
            "track_status": 0.0,
            "traffic": 0.0,
            "weather": 0.0,
        }
        pitted = action != "STAY_OUT"
        for step, status_name in enumerate(scenario.track_status_path, start=1):
            sim_state.lap = state.lap + step - 1
            sim_state.track_status = _track_status_from_name(status_name)
            sim_state.weather = dict(scenario.weather_path[step - 1])
            if pitted and step == 1:
                car.tyre_compound = _tyre_compound_from_action(action)
                car.tyre_age_laps = 0
                car.pit_out = True
                pit_loss_ms_used += _pit_loss_ms(sim_state.track_status) + 2500.0
            else:
                car.tyre_age_laps += 1
                car.pit_out = False
            car.cleaning_flags = replace(
                car.cleaning_flags,
                is_sc_vsc=sim_state.track_status in {TrackStatus.SC, TrackStatus.VSC},
                is_outlap=pitted and step == 1,
            )
            prediction = suite.pace_model.predict_lap_time(sim_state, driver_id)
            per_lap_times_ms.append(prediction.mean_ms)
            for component, value in prediction.components.items():
                component_totals_ms[component] += float(value)
            car.last_lap_time_ms = prediction.mean_ms
            if not car.cleaning_flags.is_sc_vsc and not car.cleaning_flags.is_outlap:
                recent = list(car.recent_lap_times_ms)
                recent.append(prediction.mean_ms)
                car.recent_lap_times_ms = recent[-5:]
        total_time_ms = accumulate_plan_total_time(
            per_lap_times_ms,
            pit_loss_ms=pit_loss_ms_used,
        )
        expected_total_time_ms = sum(per_lap_times_ms) + pit_loss_ms_used
        if not math.isclose(total_time_ms, expected_total_time_ms, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("distribution total time does not match per-lap accumulation")
        outcomes.append(
            ActionSimulationOutcome(
                total_time_ms=total_time_ms,
                pit_loss_ms_used=pit_loss_ms_used,
                per_lap_times_ms=per_lap_times_ms,
                pace_component_totals_ms=component_totals_ms,
            )
        )
    return outcomes


def _plan_from_distribution(
    *,
    state: RaceState,
    driver_id: str,
    plan_id: str,
    action: str,
    distribution: list[ActionSimulationOutcome],
    baseline_distribution: list[ActionSimulationOutcome],
    suite: ModelSuite,
    race_total_laps: int,
    seed: int,
    n_scenarios: int,
) -> Plan:
    plan_totals = [outcome.total_time_ms for outcome in distribution]
    baseline_totals = [outcome.total_time_ms for outcome in baseline_distribution]
    deltas = [
        compute_delta_time(plan, baseline)
        for baseline, plan in zip(baseline_totals, plan_totals, strict=True)
    ]
    mean_delta = sum(deltas) / len(deltas)
    sigma = _distribution_sigma(deltas)
    plan_total_time_ms = sum(plan_totals) / len(plan_totals)
    baseline_total_time_ms = sum(baseline_totals) / len(baseline_totals)
    plan_total_time_p50_ms = quantile(plan_totals, 0.50) or plan_total_time_ms
    baseline_total_time_p50_ms = quantile(baseline_totals, 0.50) or baseline_total_time_ms
    horizon_laps = len(distribution[0].per_lap_times_ms) if distribution else 0
    pit_loss_ms_used = sum(outcome.pit_loss_ms_used for outcome in distribution) / len(distribution)
    component_totals = _mean_component_totals(distribution)
    suspicion_reason = suspicion_reason_for_delta_time(
        delta_time_ms=mean_delta,
        horizon_laps=max(horizon_laps, 1),
    )
    counterfactuals = {
        "vs_STAY_OUT": {
            "delta_time_mean_ms": mean_delta,
            "delta_time_p50_ms": compute_delta_time(
                plan_total_time_p50_ms,
                baseline_total_time_p50_ms,
            ),
        }
    }
    return Plan(
        plan_id=plan_id,
        actions=_plan_actions(state.lap, action),
        metrics=PlanMetrics(
            delta_time_mean_ms=mean_delta,
            delta_time_p10_ms=quantile(deltas, 0.10) or mean_delta,
            delta_time_p50_ms=quantile(deltas, 0.50) or mean_delta,
            delta_time_p90_ms=quantile(deltas, 0.90) or mean_delta,
            p_gain_pos_ge_1=sum(1 for delta in deltas if delta > 0.0) / len(deltas),
            risk_sigma_ms=sigma,
        ),
        explanations=_plan_explanations(
            state=state,
            driver_id=driver_id,
            plan_id=plan_id,
            action=action,
            suite=suite,
            race_total_laps=race_total_laps,
        ),
        counterfactuals=counterfactuals,
        diagnostics={
            "horizon_laps": horizon_laps,
            "n_scenarios": n_scenarios,
            "seed": seed,
            "pit_loss_ms_used": pit_loss_ms_used,
            "per_lap_time_components_summary_ms": {
                f"{component}_mean_ms": total / horizon_laps if horizon_laps else 0.0
                for component, total in component_totals.items()
            },
            "baseline_total_time_ms": baseline_total_time_ms,
            "plan_total_time_ms": plan_total_time_ms,
            "plan_total_time_p10_ms": quantile(plan_totals, 0.10) or plan_total_time_ms,
            "plan_total_time_p50_ms": plan_total_time_p50_ms,
            "plan_total_time_p90_ms": quantile(plan_totals, 0.90) or plan_total_time_ms,
            "baseline_total_time_p50_ms": baseline_total_time_p50_ms,
            "delta_time_cap_ms": delta_time_cap_ms(horizon_laps=max(horizon_laps, 1)),
            "is_suspicious": suspicion_reason is not None,
            "suspicion_reason": suspicion_reason,
        },
        is_suspicious=suspicion_reason is not None,
        suspicion_reason=suspicion_reason,
    )


def _plan_explanations(
    *,
    state: RaceState,
    driver_id: str,
    plan_id: str,
    action: str,
    suite: ModelSuite,
    race_total_laps: int,
) -> list[Explanation]:
    car = state.cars[driver_id]
    pit_prob = suite.pit_policy_model.predict_pit_prob(state, driver_id, window_laps=3)
    reasons = [
        Explanation(
            code="TRACK_POSITION",
            text=f"{plan_id} evaluates the next {state.lap + 1} action in replay.",
            evidence={"action": action or "STAY_OUT"},
        ),
        Explanation(
            code="TYRE_CLIFF",
            text="Tyre age and recent pace feed the baseline pit/pace models.",
            evidence={"tyre_age_laps": car.tyre_age_laps, "p_pit_next_3": pit_prob.p_pit_in_window},
        ),
    ]
    laps_remaining = max(0, race_total_laps - state.lap)
    planned_compound = action.removeprefix("PIT_TO_") if action.startswith("PIT_TO_") else None
    if (
        not car.used_wet
        and len(car.used_dry_compounds) < 2
        and planned_compound is not None
        and laps_remaining <= TWO_DRY_DEADLINE_LAPS
    ):
        reasons.append(
            Explanation(
                code="RULE_COMPLIANCE",
                text="The pit action preserves dry-tyre legality before the deadline.",
                evidence={
                    "used_wet": car.used_wet,
                    "used_dry_compounds": sorted(car.used_dry_compounds),
                    "required_distinct_dry": 2,
                    "laps_remaining": laps_remaining,
                    "deadline_laps": TWO_DRY_DEADLINE_LAPS,
                    "planned_new_compound": planned_compound,
                },
            )
        )
    if state.track_status in {TrackStatus.SC, TrackStatus.VSC}:
        reasons.append(
            Explanation(
                code="SC_WINDOW",
                text="Scenario rollouts include current neutralization persistence.",
                evidence={"track_status": state.track_status.value},
            )
        )
    if len(reasons) <= 3:
        return reasons
    compliance = [reason for reason in reasons if reason.code == "RULE_COMPLIANCE"]
    if not compliance:
        return reasons[:3]
    trimmed = compliance + [reason for reason in reasons if reason.code != "RULE_COMPLIANCE"]
    return trimmed[:3]


def _plan_comparison(best_plan: Plan, baseline_plan: Plan) -> PlanComparison:
    best_total = float(best_plan.diagnostics.get("plan_total_time_ms", 0.0))
    baseline_total = float(baseline_plan.diagnostics.get("plan_total_time_ms", 0.0))
    return PlanComparison(
        baseline_plan_id=baseline_plan.plan_id,
        delta_time_mean_ms=compute_delta_time(best_total, baseline_total),
        notes="Positive means the selected plan is estimated to outperform the baseline.",
    )


def _reference_car(*, state: RaceState, driver_id: str, copy_policy: str) -> CarState | None:
    ordered = sorted(state.cars.values(), key=lambda car: (car.position, car.driver_id))
    if copy_policy == "leader":
        return ordered[0] if ordered and ordered[0].driver_id != driver_id else None
    target = state.cars[driver_id]
    for car in ordered:
        if car.position == target.position - 1:
            return car
    return ordered[0] if ordered and ordered[0].driver_id != driver_id else None


def _distribution_sigma(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _mean_component_totals(outcomes: list[ActionSimulationOutcome]) -> dict[str, float]:
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


def _scenario_seed(*, state: RaceState, driver_id: str, base_seed: int) -> int:
    return base_seed + state.lap * 1000 + sum(ord(char) for char in driver_id)


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


def _track_status_from_name(name: str) -> TrackStatus:
    normalized = name.strip().upper()
    for status in TrackStatus:
        if status.value == normalized:
            return status
    return TrackStatus.GREEN


def _pit_loss_ms(track_status: TrackStatus) -> float:
    return 12000.0 if track_status in {TrackStatus.SC, TrackStatus.VSC} else 22000.0


def _plan_actions(current_lap: int, action: str) -> list[dict[str, int | str]]:
    if action == "STAY_OUT":
        return []
    return [{"at_lap": current_lap + 1, "action": action}]


def _pit_action_for_compound(compound: TyreCompound) -> str:
    if compound is TyreCompound.SOFT:
        return "PIT_TO_MEDIUM"
    if compound is TyreCompound.MEDIUM:
        return "PIT_TO_HARD"
    if compound is TyreCompound.HARD:
        return "PIT_TO_MEDIUM"
    return "PIT_TO_HARD"


def _first_allowed_action(mask: dict[str, bool]) -> str:
    for action in ("PIT_TO_MEDIUM", "PIT_TO_HARD", "PIT_TO_SOFT", "PIT_TO_INTER", "PIT_TO_WET"):
        if mask.get(action, False):
            return action
    return "STAY_OUT"


def _tyre_compound_from_action(action: str) -> TyreCompound:
    suffix = action.removeprefix("PIT_TO_")
    for compound in TyreCompound:
        if compound.value == suffix:
            return compound
    return TyreCompound.UNKNOWN


def _actual_pit_in_window(
    *,
    pit_laps_by_driver: dict[str, list[int]],
    driver_id: str,
    current_lap: int,
    window: int,
) -> bool:
    return any(
        current_lap < lap <= current_lap + window
        for lap in pit_laps_by_driver.get(driver_id, [])
    )


def _predicted_immediate_action(plan_payload: dict[str, Any]) -> str:
    actions = plan_payload["actions"]
    if not actions:
        return "STAY_OUT"
    return "PIT"


def _predicted_compound(plan_payload: dict[str, Any]) -> str | None:
    actions = plan_payload["actions"]
    if not actions:
        return None
    return str(actions[0]["action"]).removeprefix("PIT_TO_")


def _serialize_recommendation_bundle(bundle: RecommendationBundle) -> dict[str, Any]:
    return {
        "session_id": bundle.session_id,
        "lap": bundle.lap,
        "target_driver": bundle.target_driver,
        "generated_at_ts": bundle.generated_at_ts,
        "top_k": [_serialize_plan(plan) for plan in bundle.top_k],
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


def _serialize_plan(plan: Plan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "actions": list(plan.actions),
        "metrics": {
            "delta_time_mean_ms": plan.metrics.delta_time_mean_ms,
            "delta_time_p10_ms": plan.metrics.delta_time_p10_ms,
            "delta_time_p50_ms": plan.metrics.delta_time_p50_ms,
            "delta_time_p90_ms": plan.metrics.delta_time_p90_ms,
            "p_gain_pos_ge_1": plan.metrics.p_gain_pos_ge_1,
            "risk_sigma_ms": plan.metrics.risk_sigma_ms,
        },
        "explanations": [
            {
                "code": explanation.code,
                "text": explanation.text,
                "evidence": dict(explanation.evidence),
            }
            for explanation in plan.explanations
        ],
        "counterfactuals": dict(plan.counterfactuals),
        "diagnostics": dict(plan.diagnostics),
        "is_suspicious": plan.is_suspicious,
        "suspicion_reason": plan.suspicion_reason,
    }


def _render_markdown_summary(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation Summary: {report.session_id}",
        "",
        f"- Generated at: {report.generated_at_utc}",
        f"- Feature schema version: {report.feature_schema_version}",
        f"- Assumptions hash: `{report.assumptions_hash}`",
        f"- Model versions: `{json.dumps(report.model_versions, sort_keys=True)}`",
        "",
        "## Δtime Definition",
        "",
        f"- Formula: `{DELTA_TIME_FORMULA}`",
        f"- Interpretation: {DELTA_TIME_DEFINITION_LABEL}",
        "- Units: milliseconds (`ms`)",
        "",
        "## Layer 1 Behavioral",
        "",
    ]
    for window, metrics in report.behavioral["pit_in_window"].items():
        lines.append(
            f"- {window}: AUROC={_fmt(metrics['auroc'])}, "
            f"AUPRC={_fmt(metrics['auprc'])}, "
            f"Brier={_fmt(metrics['brier'])}, n={metrics['count']}"
        )
    timing = report.behavioral["pit_timing_error"]
    lines.extend(
        [
            "",
            f"- Pit timing error median={_fmt(timing['median_abs_error_laps'])} laps, "
            f"p90={_fmt(timing['p90_abs_error_laps'])} laps, n={timing['count']}",
            f"- Top-1 action accuracy={_fmt(report.behavioral['top1_action_accuracy'])}",
            f"- Pit compound accuracy={_fmt(report.behavioral['pit_compound_accuracy'])}",
            "",
            f"- Rule violation rate={_fmt(report.decision_quality['rule_violation_rate'])}",
            "",
            "## Layer 2 Decision Quality",
            "",
        ]
    )
    for baseline_name, summary in report.decision_quality["summary_by_baseline"].items():
        lines.append(
            f"- vs {baseline_name}: mean={_fmt(summary['mean'])}, "
            f"median={_fmt(summary['median'])}, "
            f"p10={_fmt(summary['p10'])}, "
            f"p90={_fmt(summary['p90'])}, "
            f"% positive={_fmt(summary['pct_positive'])}"
        )
    lines.extend(
        [
            "",
            "## What Teams Actually Did",
            "",
        ]
    )
    for driver_id, pit_count in sorted(report.ground_truth_summary["pits_per_driver"].items()):
        compounds = report.ground_truth_summary["compound_sequence_by_driver"].get(driver_id, [])
        lines.append(
            f"- {driver_id}: pits={pit_count}, stint compounds={', '.join(compounds) or 'n/a'}"
        )
    lines.extend(
        [
            "",
            "Key pit laps:",
        ]
    )
    for item in report.ground_truth_summary["key_pit_laps"]:
        lines.append(
            f"- {item['driver_id']} lap {item['lap']}: "
            f"{item['compound_before']} -> {item['compound_after']} "
            f"({item['track_status'] or 'UNKNOWN'})"
        )
    lines.extend(
        [
            "",
            "## Pred vs Actual Sample",
            "",
            "| Driver | Lap | Predicted | Actual | Pred Compound | Actual Compound |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in _pred_vs_actual_sample(report.bundles):
        lines.append(
            f"| {sample['driver_id']} | {sample['lap']} | {sample['predicted_action']} | "
            f"{sample['actual_action']} | {sample['predicted_compound'] or '-'} | "
            f"{sample['actual_compound'] or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.3f}"
    return str(value)


def _pred_vs_actual_sample(bundle_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    if not bundle_rows:
        return []
    sample_driver = str(bundle_rows[0]["driver_id"])
    rows = [row for row in bundle_rows if row["driver_id"] == sample_driver][:5]
    sample: list[dict[str, object]] = []
    for row in rows:
        bundle = row["recommendation_bundle"]
        top_plan = bundle["top_k"][0]
        sample.append(
            {
                "driver_id": row["driver_id"],
                "lap": row["lap"],
                "predicted_action": _predicted_immediate_action(top_plan),
                "actual_action": bundle["ground_truth"]["actual_action"],
                "predicted_compound": _predicted_compound(top_plan),
                "actual_compound": bundle["ground_truth"]["actual_compound_after"],
            }
        )
    return sample
