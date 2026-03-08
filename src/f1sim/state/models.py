"""Canonical replay-first state dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TrackStatus(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    VSC = "VSC"
    SC = "SC"
    RED = "RED"


class TyreCompound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTER = "INTER"
    WET = "WET"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class CleaningFlags:
    is_inlap: bool = False
    is_outlap: bool = False
    is_sc_vsc: bool = False
    is_traffic_heavy: bool = False
    has_valid_lap_time: bool = True

    @property
    def is_clean(self) -> bool:
        return (
            self.has_valid_lap_time
            and not self.is_inlap
            and not self.is_outlap
            and not self.is_sc_vsc
            and not self.is_traffic_heavy
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "is_inlap": self.is_inlap,
            "is_outlap": self.is_outlap,
            "is_sc_vsc": self.is_sc_vsc,
            "is_traffic_heavy": self.is_traffic_heavy,
            "has_valid_lap_time": self.has_valid_lap_time,
            "is_clean": self.is_clean,
        }


@dataclass(slots=True)
class CarLapUpdate:
    driver_id: str
    team: str
    position: int
    lap_time_ms: float | None
    gap_to_leader_ms: float | None = None
    interval_ahead_ms: float | None = None
    interval_behind_ms: float | None = None
    tyre_compound: TyreCompound | str | None = None
    tyre_age_laps: int | None = None
    pit_in: bool = False
    pit_out: bool = False
    is_pitting: bool = False
    track_status: TrackStatus | str | None = None


@dataclass(slots=True)
class CarState:
    driver_id: str
    team: str
    position: int
    gap_to_leader_ms: float | None
    interval_ahead_ms: float | None
    interval_behind_ms: float | None
    tyre_compound: TyreCompound = TyreCompound.UNKNOWN
    tyre_age_laps: int = 0
    stint_id: int = 0
    used_dry_compounds: set[str] = field(default_factory=set)
    used_wet: bool = False
    recent_lap_times_ms: list[float] = field(default_factory=list)
    is_pitting: bool = False
    pit_in: bool = False
    pit_out: bool = False
    last_lap_time_ms: float | None = None
    cleaning_flags: CleaningFlags = field(default_factory=CleaningFlags)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tyre_compound"] = self.tyre_compound.value
        payload["used_dry_compounds"] = sorted(self.used_dry_compounds)
        payload["cleaning_flags"] = self.cleaning_flags.to_dict()
        return payload


@dataclass(slots=True)
class RaceState:
    session_id: str
    lap: int
    track_status: TrackStatus
    total_laps: int | None = None
    weather: dict[str, float | None] = field(default_factory=dict)
    cars: dict[str, CarState] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "lap": self.lap,
            "track_status": self.track_status.value,
            "total_laps": self.total_laps,
            "weather": dict(self.weather),
            "cars": {driver_id: car.to_dict() for driver_id, car in sorted(self.cars.items())},
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class LapEndTick:
    session_id: str
    lap: int
    car_updates: list[CarLapUpdate]
    track_status: TrackStatus | str | None = None
    total_laps: int | None = None
    weather: dict[str, float | None] = field(default_factory=dict)
