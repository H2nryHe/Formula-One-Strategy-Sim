"""Ground-truth team calls from public timing data.

Convention:
- A pit lap is defined as the lap where `pit_in == True`.
- `compound_after` is the next observed tyre compound for the driver after that pit-in lap.
- `team_calls` is an event table: one row per pit event only.
- Per-lap action labels default to `STAY_OUT`; `PIT` is set only on pit-event laps.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from f1sim.contracts import RecommendationBundle


@dataclass(slots=True)
class PitCall:
    session_id: str
    driver_id: str
    pit_lap: int
    compound_before: str | None
    compound_after: str | None
    stint_id_before: int | None
    stint_id_after: int | None
    call_type: str = "PIT"
    track_status: str | None = None
    event_time_ms: float | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ActionLabel:
    actual_action: str
    actual_compound_after: str | None = None
    actual_pit_window_label: dict[str, int] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "actual_action": self.actual_action,
            "actual_compound_after": self.actual_compound_after,
            "actual_pit_window_label": dict(self.actual_pit_window_label),
        }


def extract_pit_calls(
    laps_df: Any,
    events_df: Any = None,
) -> list[PitCall]:
    del events_df
    rows = _normalize_rows(laps_df)
    rows_by_driver = _rows_by_driver(rows)
    pit_calls: list[PitCall] = []

    for driver_id, driver_rows in rows_by_driver.items():
        stint_by_lap = _derive_stint_ids(driver_rows)
        next_compounds = _next_compound_after_lap(driver_rows)
        for row in driver_rows:
            pit_lap = int(row["lap_number"])
            if pit_lap <= 1 or not row.get("pit_in"):
                continue
            pit_calls.append(
                PitCall(
                    session_id=str(row["session_id"]),
                    driver_id=driver_id,
                    pit_lap=pit_lap,
                    compound_before=_coerce_str(row.get("tyre_compound")),
                    compound_after=next_compounds.get(pit_lap),
                    stint_id_before=stint_by_lap.get(pit_lap),
                    stint_id_after=(
                        stint_by_lap.get(pit_lap + 1)
                        if next_compounds.get(pit_lap) is not None
                        else None
                    ),
                    track_status=_coerce_str(row.get("track_status")),
                    event_time_ms=_coerce_float(row.get("lap_end_time_ms")),
                )
            )
    pit_calls.sort(key=lambda call: (call.pit_lap, call.driver_id))
    return pit_calls


def extract_lap_actions(state_or_tables: Any) -> dict[tuple[str, int], ActionLabel]:
    rows = _normalize_state_or_rows(state_or_tables)
    pit_calls = extract_pit_calls(rows)
    pit_call_index = {(call.driver_id, call.pit_lap): call for call in pit_calls}
    pit_laps_by_driver: dict[str, list[int]] = {}
    for call in pit_calls:
        pit_laps_by_driver.setdefault(call.driver_id, []).append(call.pit_lap)

    labels: dict[tuple[str, int], ActionLabel] = {}
    for row in rows:
        driver_id = str(row["driver_id"])
        lap = int(row["lap_number"])
        pit_call = pit_call_index.get((driver_id, lap))
        labels[(driver_id, lap)] = ActionLabel(
            actual_action="PIT" if pit_call is not None else "STAY_OUT",
            actual_compound_after=pit_call.compound_after if pit_call is not None else None,
            actual_pit_window_label={
                f"w{window}": int(
                    any(
                        lap < pit_lap <= lap + window
                        for pit_lap in pit_laps_by_driver.get(driver_id, [])
                    )
                )
                for window in (1, 3, 5)
            },
        )
    return labels


def materialize_team_calls(*, db_path: str, session_id: str) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_team_calls_table(conn)
        laps = conn.execute(
            """
            SELECT *
            FROM laps
            WHERE session_id = ?
            ORDER BY driver_id, lap_number
            """,
            (session_id,),
        ).fetchall()
        calls = extract_pit_calls([dict(row) for row in laps])
        conn.execute("DELETE FROM team_calls WHERE session_id = ?", (session_id,))
        conn.executemany(
            """
            INSERT INTO team_calls (
                session_id, driver_id, lap, actual_action, compound_before,
                compound_after, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    call.session_id,
                    call.driver_id,
                    call.pit_lap,
                    call.call_type,
                    call.compound_before,
                    call.compound_after,
                    json.dumps(call.to_payload(), sort_keys=True),
                )
                for call in calls
            ],
        )
        conn.commit()
    return {"team_calls": len(calls)}


def load_team_calls(*, db_path: str, session_id: str) -> list[PitCall]:
    materialize_team_calls(db_path=db_path, session_id=session_id)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_team_calls_table(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM team_calls
            WHERE session_id = ?
            ORDER BY lap, driver_id
            """,
            (session_id,),
        ).fetchall()
    return [
        PitCall(
            session_id=str(row["session_id"]),
            driver_id=str(row["driver_id"]),
            pit_lap=int(row["lap"]),
            compound_before=_coerce_str(row["compound_before"]),
            compound_after=_coerce_str(row["compound_after"]),
            stint_id_before=_payload_value(row["payload_json"], "stint_id_before"),
            stint_id_after=_payload_value(row["payload_json"], "stint_id_after"),
            call_type=str(row["actual_action"]),
            track_status=_payload_value(row["payload_json"], "track_status"),
            event_time_ms=_payload_value(row["payload_json"], "event_time_ms"),
        )
        for row in rows
    ]


def attach_ground_truth_to_bundle(
    *,
    bundle: RecommendationBundle,
    action_labels: dict[tuple[str, int], ActionLabel],
    pit_calls: list[PitCall],
) -> RecommendationBundle:
    label = action_labels.get((bundle.target_driver, bundle.lap))
    driver_pits = [call for call in pit_calls if call.driver_id == bundle.target_driver]
    bundle.ground_truth = {
        "actual_action": label.actual_action if label is not None else "STAY_OUT",
        "actual_compound_after": label.actual_compound_after if label is not None else None,
        "pit_timeline": [
            {
                "lap": call.pit_lap,
                "compound_before": call.compound_before,
                "compound_after": call.compound_after,
                "track_status": call.track_status,
            }
            for call in driver_pits
        ],
    }
    return bundle


def summarize_team_calls(pit_calls: list[PitCall]) -> dict[str, object]:
    pits_per_driver: dict[str, int] = {}
    compound_sequence_by_driver: dict[str, list[str]] = {}
    for call in pit_calls:
        pits_per_driver[call.driver_id] = pits_per_driver.get(call.driver_id, 0) + 1
        if call.compound_after is not None:
            compound_sequence_by_driver.setdefault(call.driver_id, []).append(call.compound_after)
    return {
        "pits_per_driver": pits_per_driver,
        "compound_sequence_by_driver": compound_sequence_by_driver,
        "key_pit_laps": [
            {
                "driver_id": call.driver_id,
                "lap": call.pit_lap,
                "compound_before": call.compound_before,
                "compound_after": call.compound_after,
                "track_status": call.track_status,
            }
            for call in sorted(pit_calls, key=lambda call: (call.pit_lap, call.driver_id))[:5]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize ground-truth team calls from canonical historical timing data.",
    )
    parser.add_argument("--session_id", required=True, help="Canonical session_id in SQLite.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--out", help="Optional path to write extracted team calls as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = materialize_team_calls(db_path=args.db, session_id=args.session_id)
    if args.out:
        calls = load_team_calls(db_path=args.db, session_id=args.session_id)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump([call.to_payload() for call in calls], handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _normalize_state_or_rows(state_or_tables: Any) -> list[dict[str, Any]]:
    if isinstance(state_or_tables, dict) and "laps" in state_or_tables:
        return _normalize_rows(state_or_tables["laps"])
    if (
        isinstance(state_or_tables, list)
        and state_or_tables
        and hasattr(state_or_tables[0], "cars")
    ):
        rows: list[dict[str, Any]] = []
        for state in state_or_tables:
            for driver_id, car in state.cars.items():
                rows.append(
                    {
                        "session_id": state.session_id,
                        "driver_id": driver_id,
                        "lap_number": state.lap,
                        "pit_in": car.pit_in,
                        "pit_out": car.pit_out,
                        "tyre_compound": car.tyre_compound.value,
                        "track_status": state.track_status.value,
                        "lap_end_time_ms": None,
                    }
                )
        return rows
    return _normalize_rows(state_or_tables)


def _ensure_team_calls_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS team_calls (
            session_id TEXT NOT NULL,
            driver_id TEXT NOT NULL,
            lap INTEGER NOT NULL,
            actual_action TEXT NOT NULL,
            compound_before TEXT,
            compound_after TEXT,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (session_id, driver_id, lap)
        );

        CREATE INDEX IF NOT EXISTS idx_team_calls_session_driver
            ON team_calls(session_id, driver_id, lap);
        """
    )


def _normalize_rows(rows: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(dict(row))
    normalized.sort(key=lambda row: (str(row["driver_id"]), int(row["lap_number"])))
    return normalized


def _rows_by_driver(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["driver_id"]), []).append(row)
    return grouped


def _derive_stint_ids(rows: list[dict[str, Any]]) -> dict[int, int]:
    stint_ids: dict[int, int] = {}
    stint_id = 0
    previous_compound: str | None = None
    for row in rows:
        compound = _coerce_str(row.get("tyre_compound"))
        if previous_compound is not None and compound is not None and compound != previous_compound:
            stint_id += 1
        if row.get("pit_out"):
            stint_id += 1 if previous_compound is not None else 0
        stint_ids[int(row["lap_number"])] = stint_id
        if compound is not None:
            previous_compound = compound
    return stint_ids


def _next_compound_after_lap(rows: list[dict[str, Any]]) -> dict[int, str | None]:
    next_compound_by_lap: dict[int, str | None] = {}
    for index, row in enumerate(rows):
        next_compound = None
        for future in rows[index + 1 :]:
            future_compound = _coerce_str(future.get("tyre_compound"))
            if future_compound is not None:
                next_compound = future_compound
                break
        next_compound_by_lap[int(row["lap_number"])] = next_compound
    return next_compound_by_lap


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _payload_value(payload_json: str, key: str) -> Any:
    payload = json.loads(payload_json)
    return payload.get(key)


if __name__ == "__main__":
    main()
