"""SQLite-backed replay helpers for recommendation and evaluation flows."""

from __future__ import annotations

import sqlite3
from typing import Any

from f1sim.state import CarLapUpdate, LapEndTick, RaceState, StateEngine, TyreCompound


def load_session_rows(*, db_path: str, session_id: str) -> dict[str, Any]:
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


def replay_session(*, session_id: str, session_rows: dict[str, Any]) -> list[RaceState]:
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


def replay_state_at_lap(*, db_path: str, session_id: str, lap: int) -> RaceState:
    states = replay_session(
        session_id=session_id,
        session_rows=load_session_rows(db_path=db_path, session_id=session_id),
    )
    for state in states:
        if state.lap == lap:
            return state
    raise ValueError(f"lap {lap} not found for session_id={session_id}")


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
