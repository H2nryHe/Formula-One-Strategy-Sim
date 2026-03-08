"""Cached helpers for the read-only Streamlit demo."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from f1sim.contracts import RecommendationBundle
from f1sim.replaydb import load_session_rows, replay_session
from f1sim.state import RaceState
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


def max_lap_for_session(db_path: str, session_id: str) -> int:
    states = _session_states(db_path, session_id)
    if not states:
        raise ValueError(f"session has no replayable states: {session_id}")
    return states[-1].lap


def list_drivers(db_path: str, session_id: str) -> tuple[str, ...]:
    states = _session_states(db_path, session_id)
    if not states:
        return ()
    cars = sorted(states[0].cars.values(), key=lambda car: (car.position, car.driver_id))
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
    return state, bundle


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
