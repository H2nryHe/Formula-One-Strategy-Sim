"""Lap-end replay state engine."""

from __future__ import annotations

from dataclasses import replace

from f1sim.state.models import (
    CarLapUpdate,
    CarState,
    CleaningFlags,
    LapEndTick,
    RaceState,
    TrackStatus,
    TyreCompound,
)

_DRY_COMPOUNDS = {
    TyreCompound.SOFT,
    TyreCompound.MEDIUM,
    TyreCompound.HARD,
}
_WET_COMPOUNDS = {
    TyreCompound.INTER,
    TyreCompound.WET,
}

_TRACK_STATUS_PRIORITY = {
    TrackStatus.GREEN: 0,
    TrackStatus.YELLOW: 1,
    TrackStatus.VSC: 2,
    TrackStatus.SC: 3,
    TrackStatus.RED: 4,
}


class StateEngine:
    """Advance replay state using lap-end ticks."""

    def __init__(
        self,
        *,
        clean_lap_window: int = 5,
        traffic_gap_threshold_ms: float = 1000.0,
    ) -> None:
        self.clean_lap_window = clean_lap_window
        self.traffic_gap_threshold_ms = traffic_gap_threshold_ms
        self.state: RaceState | None = None

    def step(self, lap_end_tick: LapEndTick) -> RaceState:
        previous_state = self.state
        if previous_state is None:
            previous_state = RaceState(
                session_id=lap_end_tick.session_id,
                lap=0,
                track_status=TrackStatus.GREEN,
                total_laps=lap_end_tick.total_laps,
            )
        elif previous_state.session_id != lap_end_tick.session_id:
            raise ValueError("cannot mix multiple sessions in one StateEngine")

        track_status = _resolve_track_status(
            lap_end_tick.track_status,
            lap_end_tick.car_updates,
            previous_state.track_status,
        )
        weather = dict(previous_state.weather)
        weather.update(lap_end_tick.weather)

        updated_cars = {
            driver_id: replace(
                car,
                recent_lap_times_ms=list(car.recent_lap_times_ms),
                used_dry_compounds=set(car.used_dry_compounds),
            )
            for driver_id, car in previous_state.cars.items()
        }
        warnings = list(previous_state.warnings)

        for update in lap_end_tick.car_updates:
            previous_car = previous_state.cars.get(update.driver_id)
            updated_cars[update.driver_id] = self._build_car_state(
                update=update,
                previous=previous_car,
                track_status=track_status,
            )

        ordered_cars = _normalize_running_order(updated_cars)
        tick_driver_ids = {update.driver_id for update in lap_end_tick.car_updates}
        missing_drivers = set(previous_state.cars) - tick_driver_ids
        if missing_drivers:
            warnings.append(
                f"lap {lap_end_tick.lap}: retained stale state for missing drivers "
                f"{', '.join(sorted(missing_drivers))}"
            )

        self.state = RaceState(
            session_id=lap_end_tick.session_id,
            lap=lap_end_tick.lap,
            track_status=track_status,
            total_laps=lap_end_tick.total_laps or previous_state.total_laps,
            weather=weather,
            cars=ordered_cars,
            warnings=warnings,
        )
        return self.state

    def _build_car_state(
        self,
        *,
        update: CarLapUpdate,
        previous: CarState | None,
        track_status: TrackStatus,
    ) -> CarState:
        compound = _parse_tyre_compound(update.tyre_compound, previous)
        pit_in = bool(update.pit_in)
        pit_out = bool(update.pit_out)
        compound_changed = previous is not None and compound != previous.tyre_compound
        stint_id = self._derive_stint_id(
            previous=previous,
            pit_out=pit_out,
            compound_changed=compound_changed,
        )
        tyre_age_laps = self._derive_tyre_age_laps(
            update=update,
            previous=previous,
            pit_out=pit_out,
            compound_changed=compound_changed,
        )
        flags = _build_cleaning_flags(
            update=update,
            track_status=track_status,
            traffic_gap_threshold_ms=self.traffic_gap_threshold_ms,
        )
        used_dry_compounds, used_wet = _update_compound_usage(previous=previous, compound=compound)
        recent_lap_times_ms = list(previous.recent_lap_times_ms) if previous is not None else []
        if flags.is_clean and update.lap_time_ms is not None:
            recent_lap_times_ms.append(update.lap_time_ms)
            recent_lap_times_ms = recent_lap_times_ms[-self.clean_lap_window :]

        return CarState(
            driver_id=update.driver_id,
            team=update.team,
            position=update.position,
            gap_to_leader_ms=update.gap_to_leader_ms,
            interval_ahead_ms=update.interval_ahead_ms,
            interval_behind_ms=update.interval_behind_ms,
            tyre_compound=compound,
            tyre_age_laps=tyre_age_laps,
            stint_id=stint_id,
            used_dry_compounds=used_dry_compounds,
            used_wet=used_wet,
            recent_lap_times_ms=recent_lap_times_ms,
            is_pitting=bool(update.is_pitting or pit_in),
            pit_in=pit_in,
            pit_out=pit_out,
            last_lap_time_ms=update.lap_time_ms,
            cleaning_flags=flags,
        )

    @staticmethod
    def _derive_stint_id(
        *,
        previous: CarState | None,
        pit_out: bool,
        compound_changed: bool,
    ) -> int:
        if previous is None:
            return 0
        if pit_out or compound_changed:
            return previous.stint_id + 1
        return previous.stint_id

    @staticmethod
    def _derive_tyre_age_laps(
        *,
        update: CarLapUpdate,
        previous: CarState | None,
        pit_out: bool,
        compound_changed: bool,
    ) -> int:
        if update.tyre_age_laps is not None:
            return update.tyre_age_laps
        if previous is None:
            return 0
        if pit_out or compound_changed:
            return 0
        return previous.tyre_age_laps + 1


def _parse_tyre_compound(
    value: TyreCompound | str | None,
    previous: CarState | None,
) -> TyreCompound:
    if isinstance(value, TyreCompound):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        for compound in TyreCompound:
            if compound.value == normalized:
                return compound
    if previous is not None:
        return previous.tyre_compound
    return TyreCompound.UNKNOWN


def _update_compound_usage(
    *,
    previous: CarState | None,
    compound: TyreCompound,
) -> tuple[set[str], bool]:
    used_dry_compounds = set(previous.used_dry_compounds) if previous is not None else set()
    used_wet = previous.used_wet if previous is not None else False
    if compound in _DRY_COMPOUNDS:
        used_dry_compounds.add(compound.value)
    elif compound in _WET_COMPOUNDS:
        used_wet = True
    return used_dry_compounds, used_wet


def _resolve_track_status(
    tick_status: TrackStatus | str | None,
    car_updates: list[CarLapUpdate],
    previous_status: TrackStatus,
) -> TrackStatus:
    statuses: list[TrackStatus] = []
    if tick_status is not None:
        statuses.append(_parse_track_status(tick_status))
    for update in car_updates:
        if update.track_status is not None:
            statuses.append(_parse_track_status(update.track_status))
    if not statuses:
        return previous_status
    return max(statuses, key=lambda status: _TRACK_STATUS_PRIORITY[status])


def _parse_track_status(value: TrackStatus | str) -> TrackStatus:
    if isinstance(value, TrackStatus):
        return value
    normalized = value.strip().upper()
    numeric_map = {
        "1": TrackStatus.GREEN,
        "2": TrackStatus.YELLOW,
        "4": TrackStatus.SC,
        "5": TrackStatus.RED,
        "6": TrackStatus.VSC,
        "7": TrackStatus.VSC,
    }
    if normalized in numeric_map:
        return numeric_map[normalized]
    for status in TrackStatus:
        if status.value == normalized:
            return status
    return TrackStatus.GREEN


def _build_cleaning_flags(
    *,
    update: CarLapUpdate,
    track_status: TrackStatus,
    traffic_gap_threshold_ms: float,
) -> CleaningFlags:
    traffic_gaps = [
        gap
        for gap in (update.interval_ahead_ms, update.interval_behind_ms)
        if gap is not None and gap > 0
    ]
    min_gap = min(traffic_gaps) if traffic_gaps else None
    return CleaningFlags(
        is_inlap=bool(update.pit_in),
        is_outlap=bool(update.pit_out),
        is_sc_vsc=track_status in {TrackStatus.SC, TrackStatus.VSC},
        is_traffic_heavy=min_gap is not None and min_gap <= traffic_gap_threshold_ms,
        has_valid_lap_time=update.lap_time_ms is not None and update.lap_time_ms >= 0,
    )


def _normalize_running_order(cars: dict[str, CarState]) -> dict[str, CarState]:
    ordered = sorted(cars.values(), key=lambda car: (car.position, car.driver_id))

    previous_gap: float | None = None
    for index, car in enumerate(ordered):
        if index == 0:
            car.gap_to_leader_ms = 0.0 if car.gap_to_leader_ms is None else car.gap_to_leader_ms
            car.interval_ahead_ms = 0.0
            previous_gap = car.gap_to_leader_ms
            continue

        if (
            car.gap_to_leader_ms is None
            and previous_gap is not None
            and car.interval_ahead_ms is not None
        ):
            car.gap_to_leader_ms = previous_gap + car.interval_ahead_ms
        if (
            car.interval_ahead_ms is None
            and car.gap_to_leader_ms is not None
            and previous_gap is not None
        ):
            car.interval_ahead_ms = car.gap_to_leader_ms - previous_gap
        previous_gap = car.gap_to_leader_ms if car.gap_to_leader_ms is not None else previous_gap

    for index, car in enumerate(ordered):
        if index == len(ordered) - 1:
            car.interval_behind_ms = None
            continue
        behind = ordered[index + 1]
        if behind.gap_to_leader_ms is not None and car.gap_to_leader_ms is not None:
            car.interval_behind_ms = behind.gap_to_leader_ms - car.gap_to_leader_ms
        else:
            car.interval_behind_ms = behind.interval_ahead_ms

    return {car.driver_id: car for car in ordered}
