from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from f1sim.ingest.fastf1_connector import FastF1Connector


class FakeSession:
    def __init__(self, laps: list[dict], weather_data: list[dict]) -> None:
        self.event = {"EventName": "Unit Test GP", "Location": "Synthetic Ring"}
        self.date = datetime(2023, 7, 2, 13, 0, tzinfo=timezone.utc)
        self.laps = laps
        self.weather_data = weather_data


def build_laps_fixture() -> list[dict]:
    return [
        {
            "Driver": "VER",
            "DriverNumber": "1",
            "Team": "Red Bull",
            "LapNumber": 1,
            "Position": 1,
            "LapTime": timedelta(minutes=1, seconds=31, milliseconds=200),
            "Sector1Time": timedelta(seconds=28, milliseconds=400),
            "Sector2Time": timedelta(seconds=31, milliseconds=100),
            "Sector3Time": timedelta(seconds=31, milliseconds=700),
            "Compound": "MEDIUM",
            "TyreLife": 1,
            "TrackStatus": "1",
            "IsAccurate": True,
            "PitInTime": None,
            "PitOutTime": None,
            "Time": timedelta(minutes=1, seconds=31, milliseconds=200),
        },
        {
            "Driver": "VER",
            "DriverNumber": "1",
            "Team": "Red Bull",
            "LapNumber": 2,
            "Position": 1,
            "LapTime": timedelta(minutes=1, seconds=32),
            "Sector1Time": timedelta(seconds=28, milliseconds=600),
            "Sector2Time": timedelta(seconds=31, milliseconds=400),
            "Sector3Time": timedelta(seconds=32),
            "Compound": "MEDIUM",
            "TyreLife": 2,
            "TrackStatus": "1",
            "IsAccurate": True,
            "PitInTime": timedelta(minutes=3, seconds=2),
            "PitOutTime": None,
            "Time": timedelta(minutes=3, seconds=3, milliseconds=200),
        },
        {
            "Driver": "LEC",
            "DriverNumber": "16",
            "Team": "Ferrari",
            "LapNumber": 1,
            "Position": 2,
            "LapTime": timedelta(minutes=1, seconds=31, milliseconds=900),
            "Sector1Time": timedelta(seconds=28, milliseconds=700),
            "Sector2Time": timedelta(seconds=31, milliseconds=300),
            "Sector3Time": timedelta(seconds=31, milliseconds=900),
            "Compound": "MEDIUM",
            "TyreLife": 1,
            "TrackStatus": "1",
            "IsAccurate": True,
            "PitInTime": None,
            "PitOutTime": None,
            "Time": timedelta(minutes=1, seconds=31, milliseconds=900),
        },
        {
            "Driver": "LEC",
            "DriverNumber": "16",
            "Team": "Ferrari",
            "LapNumber": 2,
            "Position": 2,
            "LapTime": timedelta(minutes=1, seconds=33),
            "Sector1Time": timedelta(seconds=28, milliseconds=800),
            "Sector2Time": timedelta(seconds=31, milliseconds=500),
            "Sector3Time": timedelta(seconds=32, milliseconds=700),
            "Compound": "MEDIUM",
            "TyreLife": 2,
            "TrackStatus": "4",
            "IsAccurate": True,
            "PitInTime": None,
            "PitOutTime": timedelta(minutes=3, seconds=4),
            "Time": timedelta(minutes=3, seconds=4, milliseconds=900),
        },
    ]


def build_weather_fixture() -> list[dict]:
    return [
        {
            "Time": timedelta(minutes=1),
            "AirTemp": 24.5,
            "TrackTemp": 33.1,
            "Humidity": 52.0,
            "Pressure": 1008.0,
            "Rainfall": 0.0,
            "WindSpeed": 1.5,
            "WindDirection": 180.0,
        },
        {
            "Time": timedelta(minutes=2),
            "AirTemp": 24.2,
            "TrackTemp": 32.8,
            "Humidity": 53.0,
            "Pressure": 1008.2,
            "Rainfall": 0.0,
            "WindSpeed": 1.8,
            "WindDirection": 185.0,
        },
    ]


def test_ingest_creates_schema_and_inserts_rows(tmp_path) -> None:
    db_path = tmp_path / "ingest.sqlite"
    fake_session = FakeSession(build_laps_fixture(), build_weather_fixture())
    connector = FastF1Connector(session_loader=lambda year, gp, session: fake_session)

    summary = connector.load_session(
        year=2023,
        gp="Unit Test GP",
        session="R",
        db_path=str(db_path),
    )

    assert summary.session_id == "2023_unit_test_gp_r"
    assert summary.sessions == 1
    assert summary.cars == 2
    assert summary.laps == 4
    assert summary.events == 4
    assert summary.weather == 2

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert {"sessions", "cars", "laps", "events", "weather"} <= tables

        counts = {
            "sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "cars": conn.execute("SELECT COUNT(*) FROM cars").fetchone()[0],
            "laps": conn.execute("SELECT COUNT(*) FROM laps").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "weather": conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0],
        }
        assert counts == {"sessions": 1, "cars": 2, "laps": 4, "events": 4, "weather": 2}

        ver_gap = conn.execute(
            """
            SELECT gap_to_leader_ms, interval_ahead_ms
            FROM laps
            WHERE session_id = ? AND driver_id = ? AND lap_number = ?
            """,
            ("2023_unit_test_gp_r", "LEC", 1),
        ).fetchone()
        assert ver_gap == pytest.approx((700.0, 700.0))


def test_ingest_rejects_negative_lap_times(tmp_path) -> None:
    db_path = tmp_path / "negative.sqlite"
    laps = build_laps_fixture()
    laps[0]["LapTime"] = timedelta(milliseconds=-1)
    connector = FastF1Connector(session_loader=lambda year, gp, session: FakeSession(laps, []))

    with pytest.raises(ValueError, match="negative lap time"):
        connector.load_session(year=2023, gp="Unit Test GP", session="R", db_path=str(db_path))


def test_ingest_rejects_duplicate_lap_numbers(tmp_path) -> None:
    db_path = tmp_path / "duplicate.sqlite"
    laps = build_laps_fixture()
    laps[1]["LapNumber"] = 1
    connector = FastF1Connector(session_loader=lambda year, gp, session: FakeSession(laps, []))

    with pytest.raises(ValueError, match="strictly increasing"):
        connector.load_session(year=2023, gp="Unit Test GP", session="R", db_path=str(db_path))


def test_ingest_treats_nan_pit_timestamps_as_missing(tmp_path) -> None:
    db_path = tmp_path / "nan_pits.sqlite"
    laps = build_laps_fixture()
    laps[0]["PitInTime"] = float("nan")
    laps[0]["PitOutTime"] = float("nan")
    connector = FastF1Connector(session_loader=lambda year, gp, session: FakeSession(laps, []))

    connector.load_session(year=2023, gp="Unit Test GP", session="R", db_path=str(db_path))

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT pit_in, pit_out
            FROM laps
            WHERE session_id = ? AND driver_id = ? AND lap_number = ?
            """,
            ("2023_unit_test_gp_r", "VER", 1),
        ).fetchone()

    assert row == (0, 0)
