"""FastF1 historical ingest into the canonical SQLite schema."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f1sim.ground_truth.team_calls import materialize_team_calls

SessionLoader = Callable[[int, str, str], Any]


@dataclass(slots=True)
class IngestSummary:
    session_id: str
    sessions: int
    cars: int
    laps: int
    events: int
    weather: int


@dataclass(slots=True)
class SessionRow:
    session_id: str
    year: int
    gp: str
    session_type: str
    event_name: str | None
    circuit_name: str | None
    source: str
    start_time_utc: str | None
    lap_count: int
    loaded_at_utc: str


@dataclass(slots=True)
class CarRow:
    session_id: str
    driver_id: str
    team_name: str | None
    car_number: str | None


@dataclass(slots=True)
class LapRow:
    session_id: str
    driver_id: str
    lap_number: int
    position: int | None
    lap_time_ms: float | None
    sector1_time_ms: float | None
    sector2_time_ms: float | None
    sector3_time_ms: float | None
    gap_to_leader_ms: float | None
    interval_ahead_ms: float | None
    tyre_compound: str | None
    tyre_age_laps: int | None
    track_status: str | None
    is_accurate: int
    pit_in: int
    pit_out: int
    lap_end_time_ms: float | None


@dataclass(slots=True)
class EventRow:
    session_id: str
    event_idx: int
    driver_id: str | None
    lap_number: int | None
    event_type: str
    event_time_ms: float | None
    payload_json: str


@dataclass(slots=True)
class WeatherRow:
    session_id: str
    weather_idx: int
    sample_time_ms: float | None
    air_temp_c: float | None
    track_temp_c: float | None
    humidity: float | None
    pressure: float | None
    rainfall: float | None
    wind_speed_ms: float | None
    wind_direction_deg: float | None


class FastF1Connector:
    """Load a FastF1 session and persist canonical tables into SQLite."""

    def __init__(self, session_loader: SessionLoader | None = None) -> None:
        self._session_loader = session_loader or self._default_session_loader

    def load_session(self, *, year: int, gp: str, session: str, db_path: str) -> IngestSummary:
        source_session = self._session_loader(year, gp, session)
        normalized = self._normalize_session(
            source_session=source_session,
            year=year,
            gp=gp,
            session=session,
        )

        db_file = Path(db_path)
        if db_file.parent != Path():
            db_file.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(db_file) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            self.create_schema(conn)
            self._delete_existing_session(conn, normalized["session"].session_id)
            self._insert_session(conn, normalized)
            conn.commit()
        materialize_team_calls(db_path=str(db_file), session_id=normalized["session"].session_id)

        return IngestSummary(
            session_id=normalized["session"].session_id,
            sessions=1,
            cars=len(normalized["cars"]),
            laps=len(normalized["laps"]),
            events=len(normalized["events"]),
            weather=len(normalized["weather"]),
        )

    @staticmethod
    def create_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                year INTEGER NOT NULL,
                gp TEXT NOT NULL,
                session_type TEXT NOT NULL,
                event_name TEXT,
                circuit_name TEXT,
                source TEXT NOT NULL,
                start_time_utc TEXT,
                lap_count INTEGER NOT NULL,
                loaded_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cars (
                session_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                team_name TEXT,
                car_number TEXT,
                PRIMARY KEY (session_id, driver_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS laps (
                session_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                lap_number INTEGER NOT NULL,
                position INTEGER,
                lap_time_ms REAL,
                sector1_time_ms REAL,
                sector2_time_ms REAL,
                sector3_time_ms REAL,
                gap_to_leader_ms REAL,
                interval_ahead_ms REAL,
                tyre_compound TEXT,
                tyre_age_laps INTEGER,
                track_status TEXT,
                is_accurate INTEGER NOT NULL,
                pit_in INTEGER NOT NULL,
                pit_out INTEGER NOT NULL,
                lap_end_time_ms REAL,
                PRIMARY KEY (session_id, driver_id, lap_number),
                FOREIGN KEY (session_id, driver_id) REFERENCES cars(session_id, driver_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                session_id TEXT NOT NULL,
                event_idx INTEGER NOT NULL,
                driver_id TEXT,
                lap_number INTEGER,
                event_type TEXT NOT NULL,
                event_time_ms REAL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, event_idx),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS weather (
                session_id TEXT NOT NULL,
                weather_idx INTEGER NOT NULL,
                sample_time_ms REAL,
                air_temp_c REAL,
                track_temp_c REAL,
                humidity REAL,
                pressure REAL,
                rainfall REAL,
                wind_speed_ms REAL,
                wind_direction_deg REAL,
                PRIMARY KEY (session_id, weather_idx),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS team_calls (
                session_id TEXT NOT NULL,
                driver_id TEXT NOT NULL,
                lap INTEGER NOT NULL,
                actual_action TEXT NOT NULL,
                compound_before TEXT,
                compound_after TEXT,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, driver_id, lap),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_laps_session_driver
                ON laps(session_id, driver_id, lap_number);
            CREATE INDEX IF NOT EXISTS idx_events_session_type
                ON events(session_id, event_type, lap_number);
            CREATE INDEX IF NOT EXISTS idx_weather_session
                ON weather(session_id, weather_idx);
            CREATE INDEX IF NOT EXISTS idx_team_calls_session_driver
                ON team_calls(session_id, driver_id, lap);
            """
        )

    @staticmethod
    def _default_session_loader(year: int, gp: str, session: str) -> Any:
        try:
            import fastf1
        except ImportError as exc:
            raise RuntimeError(
                "FastF1 is required for live ingestion. Install the package dependency first."
            ) from exc

        source_session = fastf1.get_session(year, gp, session)
        source_session.load(laps=True, telemetry=False, weather=True, messages=False)
        return source_session

    def _normalize_session(
        self,
        *,
        source_session: Any,
        year: int,
        gp: str,
        session: str,
    ) -> dict[str, Any]:
        session_id = _build_session_id(year=year, gp=gp, session=session)
        lap_records = _records_from_table(getattr(source_session, "laps", []))
        weather_records = _records_from_table(getattr(source_session, "weather_data", []))

        session_row = SessionRow(
            session_id=session_id,
            year=year,
            gp=gp,
            session_type=session,
            event_name=_resolve_event_name(source_session) or gp,
            circuit_name=_resolve_circuit_name(source_session),
            source="fastf1",
            start_time_utc=_to_iso8601(getattr(source_session, "date", None)),
            lap_count=max((_coerce_int(record.get("LapNumber")) or 0) for record in lap_records)
            if lap_records
            else 0,
            loaded_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        cars = self._normalize_cars(session_id=session_id, lap_records=lap_records)
        laps = self._normalize_laps(session_id=session_id, lap_records=lap_records)
        self._validate_laps(laps)
        events = self._normalize_events(session_id=session_id, laps=laps)
        weather = self._normalize_weather(session_id=session_id, weather_records=weather_records)

        return {
            "session": session_row,
            "cars": cars,
            "laps": laps,
            "events": events,
            "weather": weather,
        }

    @staticmethod
    def _normalize_cars(*, session_id: str, lap_records: list[dict[str, Any]]) -> list[CarRow]:
        cars_by_driver: dict[str, CarRow] = {}
        for record in lap_records:
            driver_id = _coerce_str(record.get("Driver"))
            if not driver_id:
                continue
            if driver_id in cars_by_driver:
                continue
            cars_by_driver[driver_id] = CarRow(
                session_id=session_id,
                driver_id=driver_id,
                team_name=_coerce_str(record.get("Team")),
                car_number=_coerce_str(record.get("DriverNumber")),
            )
        return sorted(cars_by_driver.values(), key=lambda row: row.driver_id)

    @staticmethod
    def _normalize_laps(*, session_id: str, lap_records: list[dict[str, Any]]) -> list[LapRow]:
        laps: list[LapRow] = []
        for record in lap_records:
            driver_id = _coerce_str(record.get("Driver"))
            lap_number = _coerce_int(record.get("LapNumber"))
            if not driver_id or lap_number is None:
                continue
            laps.append(
                LapRow(
                    session_id=session_id,
                    driver_id=driver_id,
                    lap_number=lap_number,
                    position=_coerce_int(record.get("Position")),
                    lap_time_ms=_to_ms(record.get("LapTime")),
                    sector1_time_ms=_to_ms(record.get("Sector1Time")),
                    sector2_time_ms=_to_ms(record.get("Sector2Time")),
                    sector3_time_ms=_to_ms(record.get("Sector3Time")),
                    gap_to_leader_ms=None,
                    interval_ahead_ms=None,
                    tyre_compound=_coerce_str(record.get("Compound")),
                    tyre_age_laps=_coerce_int(record.get("TyreLife")),
                    track_status=_coerce_str(record.get("TrackStatus")),
                    is_accurate=1 if _coerce_bool(record.get("IsAccurate"), default=True) else 0,
                    pit_in=1 if _has_value(record.get("PitInTime")) else 0,
                    pit_out=1 if _has_value(record.get("PitOutTime")) else 0,
                    lap_end_time_ms=_to_ms(record.get("Time")),
                )
            )

        laps.sort(key=lambda row: (row.driver_id, row.lap_number))
        _attach_gap_metrics(laps)
        return laps

    @staticmethod
    def _normalize_events(*, session_id: str, laps: list[LapRow]) -> list[EventRow]:
        events: list[EventRow] = []
        event_idx = 0

        statuses_by_lap: dict[int, set[str]] = {}
        for lap in laps:
            if lap.track_status:
                statuses_by_lap.setdefault(lap.lap_number, set()).add(lap.track_status)

        previous_status: str | None = None
        for lap_number in sorted(statuses_by_lap):
            track_status = max(
                statuses_by_lap[lap_number],
                key=_track_status_sort_key,
            )
            if track_status != previous_status:
                events.append(
                    EventRow(
                        session_id=session_id,
                        event_idx=event_idx,
                        driver_id=None,
                        lap_number=lap_number,
                        event_type="TRACK_STATUS",
                        event_time_ms=None,
                        payload_json=json.dumps({"track_status": track_status}, sort_keys=True),
                    )
                )
                event_idx += 1
                previous_status = track_status

        for lap in sorted(laps, key=lambda row: (row.lap_number, row.driver_id)):
            if lap.pit_in:
                events.append(
                    EventRow(
                        session_id=session_id,
                        event_idx=event_idx,
                        driver_id=lap.driver_id,
                        lap_number=lap.lap_number,
                        event_type="PIT_IN",
                        event_time_ms=lap.lap_end_time_ms,
                        payload_json=json.dumps({"lap_number": lap.lap_number}, sort_keys=True),
                    )
                )
                event_idx += 1
            if lap.pit_out:
                events.append(
                    EventRow(
                        session_id=session_id,
                        event_idx=event_idx,
                        driver_id=lap.driver_id,
                        lap_number=lap.lap_number,
                        event_type="PIT_OUT",
                        event_time_ms=lap.lap_end_time_ms,
                        payload_json=json.dumps({"lap_number": lap.lap_number}, sort_keys=True),
                    )
                )
                event_idx += 1
        return events

    @staticmethod
    def _normalize_weather(
        *,
        session_id: str,
        weather_records: list[dict[str, Any]],
    ) -> list[WeatherRow]:
        weather_rows: list[WeatherRow] = []
        for idx, record in enumerate(weather_records):
            weather_rows.append(
                WeatherRow(
                    session_id=session_id,
                    weather_idx=idx,
                    sample_time_ms=_to_ms(record.get("Time")),
                    air_temp_c=_coerce_float(record.get("AirTemp")),
                    track_temp_c=_coerce_float(record.get("TrackTemp")),
                    humidity=_coerce_float(record.get("Humidity")),
                    pressure=_coerce_float(record.get("Pressure")),
                    rainfall=_coerce_float(record.get("Rainfall")),
                    wind_speed_ms=_coerce_float(record.get("WindSpeed")),
                    wind_direction_deg=_coerce_float(record.get("WindDirection")),
                )
            )
        return weather_rows

    @staticmethod
    def _validate_laps(laps: list[LapRow]) -> None:
        laps_by_driver: dict[str, list[LapRow]] = {}
        for lap in laps:
            laps_by_driver.setdefault(lap.driver_id, []).append(lap)
            if lap.lap_time_ms is not None and lap.lap_time_ms < 0:
                raise ValueError(
                    f"negative lap time detected for driver={lap.driver_id} lap={lap.lap_number}"
                )

        for driver_id, driver_laps in laps_by_driver.items():
            previous_lap: int | None = None
            for lap in driver_laps:
                if previous_lap is not None and lap.lap_number <= previous_lap:
                    raise ValueError(
                        f"lap numbers are not strictly increasing for driver={driver_id}"
                    )
                previous_lap = lap.lap_number

    @staticmethod
    def _delete_existing_session(conn: sqlite3.Connection, session_id: str) -> None:
        conn.execute("DELETE FROM team_calls WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM weather WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM laps WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM cars WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    @staticmethod
    def _insert_session(conn: sqlite3.Connection, normalized: dict[str, Any]) -> None:
        session_row: SessionRow = normalized["session"]
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, year, gp, session_type, event_name, circuit_name,
                source, start_time_utc, lap_count, loaded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_row.session_id,
                session_row.year,
                session_row.gp,
                session_row.session_type,
                session_row.event_name,
                session_row.circuit_name,
                session_row.source,
                session_row.start_time_utc,
                session_row.lap_count,
                session_row.loaded_at_utc,
            ),
        )
        conn.executemany(
            """
            INSERT INTO cars (session_id, driver_id, team_name, car_number)
            VALUES (?, ?, ?, ?)
            """,
            [
                (row.session_id, row.driver_id, row.team_name, row.car_number)
                for row in normalized["cars"]
            ],
        )
        conn.executemany(
            """
            INSERT INTO laps (
                session_id, driver_id, lap_number, position, lap_time_ms,
                sector1_time_ms, sector2_time_ms, sector3_time_ms,
                gap_to_leader_ms, interval_ahead_ms, tyre_compound,
                tyre_age_laps, track_status, is_accurate, pit_in, pit_out, lap_end_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.session_id,
                    row.driver_id,
                    row.lap_number,
                    row.position,
                    row.lap_time_ms,
                    row.sector1_time_ms,
                    row.sector2_time_ms,
                    row.sector3_time_ms,
                    row.gap_to_leader_ms,
                    row.interval_ahead_ms,
                    row.tyre_compound,
                    row.tyre_age_laps,
                    row.track_status,
                    row.is_accurate,
                    row.pit_in,
                    row.pit_out,
                    row.lap_end_time_ms,
                )
                for row in normalized["laps"]
            ],
        )
        conn.executemany(
            """
            INSERT INTO events (
                session_id, event_idx, driver_id, lap_number,
                event_type, event_time_ms, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.session_id,
                    row.event_idx,
                    row.driver_id,
                    row.lap_number,
                    row.event_type,
                    row.event_time_ms,
                    row.payload_json,
                )
                for row in normalized["events"]
            ],
        )
        conn.executemany(
            """
            INSERT INTO weather (
                session_id, weather_idx, sample_time_ms, air_temp_c, track_temp_c,
                humidity, pressure, rainfall, wind_speed_ms, wind_direction_deg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.session_id,
                    row.weather_idx,
                    row.sample_time_ms,
                    row.air_temp_c,
                    row.track_temp_c,
                    row.humidity,
                    row.pressure,
                    row.rainfall,
                    row.wind_speed_ms,
                    row.wind_direction_deg,
                )
                for row in normalized["weather"]
            ],
        )


def _build_session_id(*, year: int, gp: str, session: str) -> str:
    return f"{year}_{_slugify(gp)}_{_slugify(session)}"


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part)


def _resolve_event_name(source_session: Any) -> str | None:
    event = getattr(source_session, "event", None)
    if event is None:
        return None
    return _coerce_str(_mapping_get(event, "EventName")) or _coerce_str(
        _mapping_get(event, "OfficialEventName")
    )


def _resolve_circuit_name(source_session: Any) -> str | None:
    event = getattr(source_session, "event", None)
    if event is None:
        return None
    return _coerce_str(_mapping_get(event, "Location")) or _coerce_str(
        _mapping_get(event, "Circuit")
    )


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _records_from_table(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_dict"):
        return list(table.to_dict("records"))
    if isinstance(table, Iterable) and not isinstance(table, (str, bytes, dict)):
        return [dict(record) for record in table]
    raise TypeError(
        "expected a table-like object with to_dict('records') or an iterable of mappings"
    )


def _attach_gap_metrics(laps: list[LapRow]) -> None:
    laps_by_number: dict[int, list[LapRow]] = {}
    for lap in laps:
        laps_by_number.setdefault(lap.lap_number, []).append(lap)

    for lap_number in sorted(laps_by_number):
        ordered = sorted(
            laps_by_number[lap_number],
            key=lambda row: (
                row.position is None,
                row.position if row.position is not None else 999,
                row.lap_end_time_ms if row.lap_end_time_ms is not None else float("inf"),
                row.driver_id,
            ),
        )
        leader_time = next(
            (row.lap_end_time_ms for row in ordered if row.lap_end_time_ms is not None),
            None,
        )
        previous_time: float | None = None
        for row in ordered:
            if leader_time is not None and row.lap_end_time_ms is not None:
                row.gap_to_leader_ms = row.lap_end_time_ms - leader_time
                row.interval_ahead_ms = (
                    None if previous_time is None else row.lap_end_time_ms - previous_time
                )
                previous_time = row.lap_end_time_ms


def _to_iso8601(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc).isoformat()
    return _coerce_str(value)


def _to_ms(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds() * 1000.0)
    coerced = _coerce_float(value)
    if coerced is None:
        return None
    return coerced


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    text = _coerce_str(value)
    if text is None:
        return False
    return text.lower() not in {"nat", "nan", "none"}


def _track_status_sort_key(status: str) -> tuple[int, str]:
    try:
        return int(status), status
    except ValueError:
        return -1, status
