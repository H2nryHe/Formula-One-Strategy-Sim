from f1sim.contracts import Plan, PlanMetrics
from f1sim.replay import build_bootstrap_state
from f1sim.rules import action_mask, compounds_used_count, is_two_dry_required, plan_satisfies_rules


def test_plan_satisfies_rules_with_second_dry_compound() -> None:
    state = build_bootstrap_state()
    state.lap = 50
    state.total_laps = 53
    car = state.cars["VER"]
    car.tyre_compound = car.tyre_compound.SOFT
    car.used_dry_compounds = {"SOFT"}
    car.used_wet = False

    compliant_plan = Plan(
        plan_id="PIT_TO_MEDIUM",
        actions=[{"at_lap": 51, "action": "PIT_TO_MEDIUM"}],
        metrics=PlanMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    non_compliant_plan = Plan(
        plan_id="STAY_OUT",
        actions=[],
        metrics=PlanMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    assert is_two_dry_required(state, "VER") is True
    assert compounds_used_count(state, "VER") == 1
    assert plan_satisfies_rules(state, "VER", compliant_plan, 53) is True
    assert plan_satisfies_rules(state, "VER", non_compliant_plan, 53) is True


def test_action_mask_forces_second_dry_compound_near_deadline() -> None:
    state = build_bootstrap_state()
    state.lap = 48
    state.total_laps = 53
    car = state.cars["VER"]
    car.tyre_compound = car.tyre_compound.SOFT
    car.used_dry_compounds = {"SOFT"}
    car.used_wet = False

    mask = action_mask(state, "VER", state.lap, state.total_laps, deadline_laps=12)

    assert mask["STAY_OUT"] is False
    assert mask["PIT_TO_SOFT"] is False
    assert mask["PIT_TO_MEDIUM"] is True
    assert mask["PIT_TO_HARD"] is True
