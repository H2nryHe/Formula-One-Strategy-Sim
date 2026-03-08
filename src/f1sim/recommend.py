"""CLI entrypoint for rollout-based strategy recommendations."""

from __future__ import annotations

import argparse
import json

from f1sim.ground_truth import attach_ground_truth_to_bundle, extract_lap_actions, load_team_calls
from f1sim.replaydb import load_session_rows, replay_state_at_lap
from f1sim.strategy import (
    RolloutSearchConfig,
    RolloutStrategySearcher,
    build_model_suite,
    recommendation_bundle_to_dict,
    validate_recommendation_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rollout-based top-K strategy recommendations from replay state.",
    )
    parser.add_argument("--session_id", required=True, help="Canonical session_id in SQLite.")
    parser.add_argument("--driver", required=True, help="Target driver identifier.")
    parser.add_argument("--lap", required=True, type=int, help="Lap-end replay tick to evaluate.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of plans to return.")
    parser.add_argument("--horizon", type=int, default=8, help="Rollout horizon in laps.")
    parser.add_argument("--n_scenarios", type=int, default=16, help="Number of sampled scenarios.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic rollout seed.")
    parser.add_argument(
        "--copy_policy",
        choices=("leader", "nearest"),
        default="nearest",
        help="Reference baseline family for RecommendationBundle baselines.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = replay_state_at_lap(
        db_path=args.db,
        session_id=args.session_id,
        lap=args.lap,
    )
    config = RolloutSearchConfig(
        horizon_laps=args.horizon,
        n_scenarios=args.n_scenarios,
        copy_policy=args.copy_policy,
        top_k=args.top_k,
    )
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(config.rule_thresholds),
        config=config,
    )
    bundle = searcher.recommend(
        state,
        args.driver,
        horizon_laps=args.horizon,
        top_k=args.top_k,
        seed=args.seed,
    )
    session_rows = load_session_rows(db_path=args.db, session_id=args.session_id)
    pit_calls = load_team_calls(db_path=args.db, session_id=args.session_id)
    action_labels = extract_lap_actions(session_rows)
    attach_ground_truth_to_bundle(
        bundle=bundle,
        action_labels=action_labels,
        pit_calls=pit_calls,
    )
    validate_recommendation_bundle(bundle, expected_top_k=min(args.top_k, len(bundle.top_k)))
    print(json.dumps(recommendation_bundle_to_dict(bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
