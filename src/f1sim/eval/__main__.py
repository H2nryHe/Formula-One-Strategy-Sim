from __future__ import annotations

import argparse

from f1sim.eval.report import write_evaluation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run replay-first evaluation for one ingested historical session.",
    )
    parser.add_argument("--session_id", required=True, help="Canonical session_id in SQLite.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--out_dir", required=True, help="Directory for JSON and markdown outputs.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Decision-quality rollout horizon in laps.",
    )
    parser.add_argument(
        "--copy_policy",
        choices=("leader", "nearest"),
        default="nearest",
        help="Imitation baseline family for Layer 2.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic base seed.")
    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=8,
        help="Number of sampled scenarios for Layer 2 rollouts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_evaluation_outputs(
        session_id=args.session_id,
        db_path=args.db,
        out_dir=args.out_dir,
        horizon_laps=args.horizon,
        copy_policy=args.copy_policy,
        seed=args.seed,
        n_scenarios=args.n_scenarios,
    )
    print(f"wrote {outputs['json_path']}")
    print(f"wrote {outputs['markdown_path']}")


if __name__ == "__main__":
    main()
