from tests.test_eval import _seed_eval_db

from f1sim.replay import build_bootstrap_state
from f1sim.replaydb import replay_state_at_lap
from f1sim.strategy import RolloutSearchConfig, RolloutStrategySearcher, build_model_suite


def test_explainer_attaches_reason_codes_and_counterfactuals(tmp_path) -> None:
    db_path = tmp_path / "explainer.sqlite"
    _seed_eval_db(str(db_path))
    state = replay_state_at_lap(
        db_path=str(db_path),
        session_id="2023_unit_test_gp_r",
        lap=4,
    )
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(horizon_laps=8, n_scenarios=8, top_k=3),
    )

    bundle = searcher.recommend(state, "VER", horizon_laps=8, top_k=3, seed=11)
    best_plan = bundle.top_k[0]
    codes = {explanation.code for explanation in best_plan.explanations}

    assert len(best_plan.explanations) >= 2
    assert "TRACK_POSITION" in codes
    assert "TYRE_CLIFF" in codes
    assert "vs_STAY_OUT" in best_plan.counterfactuals
    assert "vs_PIT_NEXT_LAP" in best_plan.counterfactuals
    assert "reference_plan_id" in best_plan.counterfactuals["vs_PIT_NEXT_LAP"]


def test_explainer_surfaces_traffic_and_sc_window_when_present() -> None:
    state = build_bootstrap_state()
    state.track_status = state.track_status.SC
    state.cars["LEC"].cleaning_flags.is_traffic_heavy = True
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(horizon_laps=6, n_scenarios=6, top_k=3),
    )

    bundle = searcher.recommend(state, "LEC", horizon_laps=6, top_k=3, seed=5)
    best_plan = bundle.top_k[0]
    evidence_by_code = {
        explanation.code: explanation.evidence for explanation in best_plan.explanations
    }

    assert "SC_WINDOW" in evidence_by_code
    assert evidence_by_code["SC_WINDOW"]["track_status"] == "SC"
    assert "TRAFFIC_PENALTY" in evidence_by_code
    assert "interval_ahead_ms" in evidence_by_code["TRAFFIC_PENALTY"]


def test_explainer_handles_leader_without_car_ahead() -> None:
    state = build_bootstrap_state()
    searcher = RolloutStrategySearcher(
        suite=build_model_suite(),
        config=RolloutSearchConfig(horizon_laps=6, n_scenarios=6, top_k=3),
    )

    bundle = searcher.recommend(state, "VER", horizon_laps=6, top_k=3, seed=11)

    assert bundle.top_k[0].plan_id
