"""Cached helpers for the read-only Streamlit demo."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from f1sim.contracts import RecommendationBundle
from f1sim.ground_truth import attach_ground_truth_to_bundle, extract_lap_actions, extract_pit_calls
from f1sim.replaydb import load_session_rows, replay_session
from f1sim.state import CarState, RaceState
from f1sim.strategy import RolloutSearchConfig, RolloutStrategySearcher, build_model_suite


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    year: int
    gp: str
    session_type: str
    event_name: str | None
    circuit_name: str | None
    lap_count: int

    @property
    def label(self) -> str:
        return f"{self.session_id} | {self.year} {self.gp} {self.session_type}"


@lru_cache(maxsize=8)
def list_sessions(db_path: str) -> tuple[SessionInfo, ...]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT session_id, year, gp, session_type, event_name, circuit_name, lap_count
            FROM sessions
            ORDER BY year DESC, gp, session_type
            """
        ).fetchall()
    return tuple(
        SessionInfo(
            session_id=str(row["session_id"]),
            year=int(row["year"]),
            gp=str(row["gp"]),
            session_type=str(row["session_type"]),
            event_name=row["event_name"],
            circuit_name=row["circuit_name"],
            lap_count=int(row["lap_count"]),
        )
        for row in rows
    )


@lru_cache(maxsize=16)
def _session_states(db_path: str, session_id: str) -> tuple[RaceState, ...]:
    session_rows = load_session_rows(db_path=db_path, session_id=session_id)
    return tuple(replay_session(session_id=session_id, session_rows=session_rows))


@lru_cache(maxsize=32)
def _session_rows(db_path: str, session_id: str) -> dict[str, object]:
    return load_session_rows(db_path=db_path, session_id=session_id)


@lru_cache(maxsize=32)
def _session_ground_truth(
    db_path: str,
    session_id: str,
) -> tuple[dict[tuple[str, int], object], tuple[object, ...]]:
    rows = _session_rows(db_path, session_id)
    return extract_lap_actions(rows), tuple(extract_pit_calls(rows["laps"]))


def max_lap_for_session(db_path: str, session_id: str) -> int:
    states = _session_states(db_path, session_id)
    if not states:
        raise ValueError(f"session has no replayable states: {session_id}")
    return states[-1].lap


def list_drivers(db_path: str, session_id: str) -> tuple[str, ...]:
    states = _session_states(db_path, session_id)
    if not states:
        return ()
    cars = sorted(_valid_ranked_cars(states[0]), key=lambda car: (car.position, car.driver_id))
    return tuple(car.driver_id for car in cars)


@lru_cache(maxsize=128)
def build_demo_payload(
    *,
    db_path: str,
    session_id: str,
    driver_id: str,
    lap: int,
    seed: int,
    top_k: int = 3,
    horizon_laps: int = 8,
    n_scenarios: int = 16,
    copy_policy: str = "nearest",
) -> tuple[RaceState, RecommendationBundle]:
    states = _session_states(db_path, session_id)
    state = _state_at_lap(states, lap)
    config = RolloutSearchConfig(
        horizon_laps=horizon_laps,
        n_scenarios=n_scenarios,
        copy_policy=copy_policy,
        top_k=top_k,
    )
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(config.rule_thresholds),
        config=config,
    )
    bundle = searcher.recommend(
        state,
        driver_id,
        horizon_laps=horizon_laps,
        top_k=top_k,
        seed=seed,
    )
    action_labels, pit_calls = _session_ground_truth(db_path, session_id)
    attach_ground_truth_to_bundle(
        bundle=bundle,
        action_labels=action_labels,
        pit_calls=list(pit_calls),
    )
    return state, bundle


@lru_cache(maxsize=128)
def build_race_situation_panel(
    *,
    db_path: str,
    session_id: str,
    lap: int,
) -> dict[str, list[dict[str, object]]]:
    state = _state_at_lap(_session_states(db_path, session_id), lap)
    cars = _valid_ranked_cars(state)
    suite = build_model_suite()
    undercut_candidates = sorted(
        (
            {
                "driver_id": car.driver_id,
                "position": car.position,
                "gap_ahead_ms": car.interval_ahead_ms,
                "tyre_age_laps": car.tyre_age_laps,
            }
            for car in cars
            if car.position > 1 and (car.interval_ahead_ms or 999999.0) <= 2500.0
        ),
        key=lambda item: (item["gap_ahead_ms"] or 999999.0, item["position"]),
    )[:5]
    tyre_risk_ranking = sorted(
        (
            {
                "driver_id": car.driver_id,
                "compound": car.tyre_compound.value,
                "tyre_age_laps": car.tyre_age_laps,
                "cliff_risk": suite.degradation_model.predict_delta(
                    car.tyre_compound.value,
                    car.tyre_age_laps,
                    {"track_status": state.track_status.value, "weather": state.weather},
                ).cliff_risk,
            }
            for car in cars
        ),
        key=lambda item: (item["cliff_risk"] or 0.0, item["tyre_age_laps"]),
        reverse=True,
    )[:5]
    traffic_hotspots = sorted(
        (
            {
                "driver_id": car.driver_id,
                "position": car.position,
                "min_gap_ms": min(
                    gap
                    for gap in (car.interval_ahead_ms, car.interval_behind_ms)
                    if gap is not None
                )
                if any(gap is not None for gap in (car.interval_ahead_ms, car.interval_behind_ms))
                else None,
                "traffic_heavy": car.cleaning_flags.is_traffic_heavy,
            }
            for car in cars
        ),
        key=lambda item: (item["min_gap_ms"] is None, item["min_gap_ms"] or 999999.0),
    )[:5]
    return {
        "undercut_candidates": undercut_candidates,
        "tyre_risk_ranking": tyre_risk_ranking,
        "traffic_hotspots": traffic_hotspots,
    }


@lru_cache(maxsize=64)
def build_key_moments(
    *,
    db_path: str,
    session_id: str,
    driver_id: str,
    seed: int,
    top_k: int = 3,
    horizon_laps: int = 8,
    n_scenarios: int = 16,
) -> tuple[dict[str, object], ...]:
    states = _session_states(db_path, session_id)
    pit_calls = list(_session_ground_truth(db_path, session_id)[1])
    pits_by_lap: dict[int, int] = {}
    for call in pit_calls:
        pits_by_lap[call.pit_lap] = pits_by_lap.get(call.pit_lap, 0) + 1

    scored: list[dict[str, object]] = []
    previous_action: str | None = None
    previous_delta: float | None = None
    for state in states:
        bundle_state, bundle = build_demo_payload(
            db_path=db_path,
            session_id=session_id,
            driver_id=driver_id,
            lap=state.lap,
            seed=seed,
            top_k=top_k,
            horizon_laps=horizon_laps,
            n_scenarios=n_scenarios,
        )
        del bundle_state
        top_plan = bundle.top_k[0]
        predicted_action = str(top_plan.diagnostics.get("immediate_action", "STAY_OUT"))
        actual_action = bundle.ground_truth.get("actual_action", "STAY_OUT")
        score = 0.0
        reasons: list[str] = []
        if predicted_action != actual_action:
            score += 3.0
            reasons.append("pred-actual mismatch")
        if previous_action is not None and predicted_action != previous_action:
            score += 2.0
            reasons.append("recommendation shift")
        if previous_delta is not None:
            delta_jump = abs(top_plan.metrics.delta_time_mean_ms - previous_delta)
            if delta_jump >= 5000.0:
                score += min(delta_jump / 5000.0, 3.0)
                reasons.append("delta jump")
        if pits_by_lap.get(state.lap, 0):
            score += 1.0 + min(pits_by_lap[state.lap], 3)
            reasons.append("opponent pit trigger")
        previous_action = predicted_action
        previous_delta = top_plan.metrics.delta_time_mean_ms
        if score <= 0.0:
            continue
        scored.append(
            {
                "lap": state.lap,
                "score": score,
                "summary": ", ".join(reasons),
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), int(item["lap"])))
    return tuple(scored[:5])


def build_timeline_rows(
    *,
    db_path: str,
    session_id: str,
    driver_id: str,
) -> list[dict[str, object]]:
    states = _session_states(db_path, session_id)
    session_rows = _session_rows(db_path, session_id)
    pit_markers = _driver_pit_markers(session_rows, driver_id)
    return [
        {
            "lap": state.lap,
            "track_status": state.track_status.value,
            "pit_in": pit_markers.get(state.lap, {}).get("pit_in", False),
            "pit_out": pit_markers.get(state.lap, {}).get("pit_out", False),
        }
        for state in states
    ]


def _state_at_lap(states: tuple[RaceState, ...], lap: int) -> RaceState:
    for state in states:
        if state.lap == lap:
            return state
    raise ValueError(f"lap {lap} not found")


def _driver_pit_markers(
    session_rows: dict[str, object],
    driver_id: str,
) -> dict[int, dict[str, bool]]:
    markers: dict[int, dict[str, bool]] = {}
    for row in session_rows["laps"]:
        if row["driver_id"] != driver_id:
            continue
        lap_number = int(row["lap_number"])
        markers[lap_number] = {
            "pit_in": bool(row["pit_in"]),
            "pit_out": bool(row["pit_out"]),
        }
    return markers


def _valid_ranked_cars(state: RaceState) -> tuple[CarState, ...]:
    return tuple(
        car
        for car in state.cars.values()
        if car.position is not None and 1 <= int(car.position) < 900
    )
