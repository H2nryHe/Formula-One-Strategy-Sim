"""Generate deterministic replay recommendation artifacts for Monza 2023."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from f1sim.assumptions import default_assumptions_hash
from f1sim.ground_truth import attach_ground_truth_to_bundle, extract_lap_actions, load_team_calls
from f1sim.metrics import DELTA_TIME_DEFINITION_LABEL, DELTA_TIME_FORMULA
from f1sim.replaydb import load_session_rows, replay_session
from f1sim.strategy import (
    RolloutSearchConfig,
    RolloutStrategySearcher,
    build_model_suite,
    recommendation_bundle_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deterministic Monza 2023 replay recommendation artifacts.",
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--session_id", required=True, help="Canonical session id.")
    parser.add_argument("--drivers", required=True, help="Comma-separated driver ids.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic rollout seed.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of plans per lap.")
    parser.add_argument("--horizon", type=int, default=10, help="Rollout horizon in laps.")
    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=200,
        help="Number of sampled scenarios per recommendation.",
    )
    parser.add_argument(
        "--deadline_laps",
        type=int,
        default=12,
        help="Two-dry rules deadline parameter.",
    )
    parser.add_argument("--out_dir", required=True, help="Artifact output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drivers = [driver.strip().upper() for driver in args.drivers.split(",") if driver.strip()]
    session_rows = load_session_rows(db_path=args.db, session_id=args.session_id)
    replay_states = replay_session(session_id=args.session_id, session_rows=session_rows)
    action_labels = extract_lap_actions(session_rows)
    pit_calls = load_team_calls(db_path=args.db, session_id=args.session_id)

    config = RolloutSearchConfig(
        horizon_laps=args.horizon,
        n_scenarios=args.n_scenarios,
        top_k=args.top_k,
        two_dry_deadline_laps=args.deadline_laps,
    )
    suite = build_model_suite(config.rule_thresholds)
    searcher = RolloutStrategySearcher(suite=suite, config=config)

    available_drivers = set(replay_states[0].cars)
    selected_drivers = [driver for driver in drivers if driver in available_drivers]

    config_payload = {
        "session_id": args.session_id,
        "db": args.db,
        "drivers": selected_drivers,
        "seed": args.seed,
        "top_k": args.top_k,
        "horizon_laps": args.horizon,
        "n_scenarios": args.n_scenarios,
        "two_dry_deadline_laps": args.deadline_laps,
        "delta_time": {
            "formula": DELTA_TIME_FORMULA,
            "interpretation": DELTA_TIME_DEFINITION_LABEL,
            "units": "ms",
        },
        "assumptions_hash": default_assumptions_hash(),
        "model_versions": suite.model_versions(),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    session_summary = {
        "session_id": args.session_id,
        "lap_count": replay_states[-1].lap,
        "drivers": sorted(available_drivers),
        "pit_counts": dict(Counter(call.driver_id for call in pit_calls)),
        "track_status_summary": dict(Counter(state.track_status.value for state in replay_states)),
    }
    (out_dir / "session_summary.json").write_text(
        json.dumps(session_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for driver in selected_drivers:
        output_path = out_dir / f"recommendations_driver_{driver}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for state in replay_states:
                bundle = searcher.recommend(
                    state,
                    driver,
                    horizon_laps=args.horizon,
                    top_k=args.top_k,
                    seed=args.seed,
                )
                attach_ground_truth_to_bundle(
                    bundle=bundle,
                    action_labels=action_labels,
                    pit_calls=pit_calls,
                )
                payload = {
                    "lap": state.lap,
                    "driver_id": driver,
                    "recommendation_bundle": recommendation_bundle_to_dict(bundle),
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
