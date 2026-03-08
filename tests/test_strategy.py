from __future__ import annotations

import json
import subprocess
import sys

from tests.test_eval import _seed_eval_db

from f1sim.replay import build_bootstrap_state
from f1sim.rules import plan_satisfies_rules
from f1sim.strategy import (
    RolloutSearchConfig,
    RolloutStrategySearcher,
    build_model_suite,
    recommendation_bundle_to_dict,
    validate_recommendation_bundle,
)


def test_rollout_search_returns_valid_top_k_bundle() -> None:
    state = build_bootstrap_state()
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(horizon_laps=8, n_scenarios=8, top_k=3),
    )

    bundle, baselines = searcher.recommend_with_artifacts(
        state,
        "LEC",
        seed=13,
    )

    validate_recommendation_bundle(bundle, expected_top_k=3)
    assert len(bundle.top_k) == 3
    assert bundle.top_k[0].metrics.delta_time_mean_ms >= bundle.top_k[1].metrics.delta_time_mean_ms
    assert set(bundle.baselines) == {"STAY_OUT", "RULE_TYRE_AGE", "COPY_NEAREST"}
    assert baselines["STAY_OUT"].plan_id == "STAY_OUT"
    assert all(len(plan.explanations) >= 2 for plan in bundle.top_k)


def test_rollout_search_is_deterministic_for_same_seed() -> None:
    state = build_bootstrap_state()
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(horizon_laps=8, n_scenarios=10, top_k=3),
    )

    bundle_a = searcher.recommend(state, "LEC", horizon_laps=8, top_k=3, seed=19)
    bundle_b = searcher.recommend(state, "LEC", horizon_laps=8, top_k=3, seed=19)

    assert recommendation_bundle_to_dict(bundle_a) == recommendation_bundle_to_dict(bundle_b)


def test_recommend_cli_outputs_bundle_json(tmp_path) -> None:
    db_path = tmp_path / "recommend.sqlite"
    _seed_eval_db(str(db_path))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "f1sim.recommend",
            "--session_id",
            "2023_unit_test_gp_r",
            "--driver",
            "VER",
            "--lap",
            "4",
            "--db",
            str(db_path),
            "--top_k",
            "3",
            "--seed",
            "11",
        ],
        cwd="/Users/linruihe/Local Documents/Github/Formula One Strategy Sim",
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["session_id"] == "2023_unit_test_gp_r"
    assert payload["target_driver"] == "VER"
    assert len(payload["top_k"]) == 3
    assert payload["model_versions"]["search"] == "rollout_search_v0.3"
    assert payload["ground_truth"]["actual_action"] in {"STAY_OUT", "PIT"}


def test_rollout_search_enforces_two_dry_rule_deadline() -> None:
    state = build_bootstrap_state()
    state.lap = 48
    state.total_laps = 53
    target = state.cars["VER"]
    target.tyre_compound = target.tyre_compound.SOFT
    target.used_dry_compounds = {"SOFT"}
    target.used_wet = False
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(
            horizon_laps=5,
            n_scenarios=6,
            top_k=3,
            two_dry_deadline_laps=12,
        ),
    )

    bundle = searcher.recommend(state, "VER", horizon_laps=5, top_k=3, seed=23)

    assert bundle.top_k
    assert all(plan.plan_id in {"PIT_TO_MEDIUM", "PIT_TO_HARD"} for plan in bundle.top_k)
    assert all(
        plan_satisfies_rules(state, "VER", plan, state.total_laps or 53) for plan in bundle.top_k
    )
    assert any(
        explanation.code == "RULE_COMPLIANCE"
        for explanation in bundle.top_k[0].explanations
    )
