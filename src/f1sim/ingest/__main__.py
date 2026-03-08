from __future__ import annotations

import argparse

from f1sim.ingest.fastf1_connector import FastF1Connector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a historical FastF1 session into the canonical SQLite schema."
    )
    parser.add_argument("--year", required=True, type=int, help="Season year, for example 2023.")
    parser.add_argument("--gp", required=True, help="Grand Prix name understood by FastF1.")
    parser.add_argument(
        "--session",
        required=True,
        help="Session code or name understood by FastF1, for example R or Q.",
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = FastF1Connector().load_session(
        year=args.year,
        gp=args.gp,
        session=args.session,
        db_path=args.db,
    )
    print(
        "ingested "
        f"session_id={summary.session_id} "
        f"sessions={summary.sessions} "
        f"cars={summary.cars} "
        f"laps={summary.laps} "
        f"events={summary.events} "
        f"weather={summary.weather}"
    )


if __name__ == "__main__":
    main()
